"""Dagster orchestration for quarterly BDC data refresh.

Lighter than Airflow on a 16GB machine. One job with assets for each pipeline step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from dagster import (
        asset,
        define_asset_job,
        Definitions,
        ScheduleDefinition,
        AssetSelection,
    )
    HAS_DAGSTER = True
except ImportError:
    HAS_DAGSTER = False
    logger.info("Dagster not installed. Orchestration will run as plain Python scripts.")


if HAS_DAGSTER:

    @asset(group_name="ingestion")
    def raw_sec_filings() -> dict:
        """Download SOI data from SEC EDGAR for all known BDCs."""
        from src.ingest import EdgarClient, ingest_bdc, KNOWN_BDCS, get_user_agent

        client = EdgarClient(get_user_agent())
        output_dir = Path("data/raw")
        total = 0
        for bdc in KNOWN_BDCS:
            count = ingest_bdc(client, bdc, output_dir)
            total += count
        return {"total_records": total, "bdcs": len(KNOWN_BDCS)}

    @asset(group_name="ingestion", deps=[raw_sec_filings])
    def sample_fixture() -> dict:
        """Create sample fixture from downloaded data."""
        from src.ingest import create_sample_fixture
        create_sample_fixture(Path("data/raw"), Path("data/sample"))
        return {"status": "created"}

    @asset(group_name="transform", deps=[raw_sec_filings])
    def load_raw_to_duckdb() -> dict:
        """Load raw JSON data into DuckDB warehouse."""
        import duckdb
        db_path = Path("data/processed/warehouse.duckdb")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))

        all_records = []
        raw_dir = Path("data/raw")
        for bdc_dir in sorted(raw_dir.iterdir()):
            records_file = bdc_dir / "soi_records.json"
            if records_file.exists():
                records = json.loads(records_file.read_text())
                all_records.extend(records)

        if all_records:
            con.execute("DROP TABLE IF EXISTS raw_soi_positions")
            con.execute("""
                CREATE TABLE raw_soi_positions AS
                SELECT * FROM read_json_auto(?)
            """, [json.dumps(all_records)])

        count = con.execute("SELECT COUNT(*) FROM raw_soi_positions").fetchone()[0]
        con.close()
        return {"records_loaded": count}

    @asset(group_name="transform", deps=[load_raw_to_duckdb])
    def entity_resolution_map() -> dict:
        """Run entity resolution and save canonical mapping."""
        from src.entity_resolution import resolve_and_save
        db_path = Path("data/processed/warehouse.duckdb")
        mapping = resolve_and_save(db_path)
        return {"canonical_borrowers": len(set(mapping.values()))}

    @asset(group_name="transform", deps=[entity_resolution_map])
    def dbt_models() -> dict:
        """Run dbt models to build staging, intermediate, and marts."""
        import subprocess
        result = subprocess.run(
            ["dbt", "run", "--project-dir", "transforms", "--profiles-dir", "transforms"],
            capture_output=True, text=True,
        )
        return {"stdout": result.stdout[-500:], "returncode": result.returncode}

    @asset(group_name="transform", deps=[dbt_models])
    def dbt_tests() -> dict:
        """Run dbt tests."""
        import subprocess
        result = subprocess.run(
            ["dbt", "test", "--project-dir", "transforms", "--profiles-dir", "transforms"],
            capture_output=True, text=True,
        )
        return {"stdout": result.stdout[-500:], "returncode": result.returncode}

    @asset(group_name="model", deps=[dbt_models])
    def early_warning_model() -> dict:
        """Train the early warning LightGBM model."""
        import duckdb
        from src.model import prepare_model_data, train_baseline, train_main_model

        db_path = Path("data/processed/warehouse.duckdb")
        con = duckdb.connect(str(db_path), read_only=True)
        df = prepare_model_data(con)
        con.close()

        if df.empty:
            return {"status": "no data for modeling"}

        quarters = sorted(df["quarter"].unique())
        if len(quarters) < 3:
            return {"status": "not enough quarters for time-based split"}

        train_end = quarters[-3]
        val_q = quarters[-2]
        test_q = quarters[-1]

        from src.model import time_based_split
        train, val, test = time_based_split(df, train_end, val_q, test_q)

        baseline = train_baseline(train, val)
        main = train_main_model(train, val, Path("models"))

        return {"baseline": baseline, "main": main}

    quarterly_refresh = define_asset_job(
        name="quarterly_refresh",
        selection=AssetSelection.all(),
    )

    quarterly_schedule = ScheduleDefinition(
        job=quarterly_refresh,
        cron_schedule="0 6 1 1,4,7,10 *",  # 6 AM on 1st of Jan, Apr, Jul, Oct
    )

    defs = Definitions(
        assets=[
            raw_sec_filings,
            sample_fixture,
            load_raw_to_duckdb,
            entity_resolution_map,
            dbt_models,
            dbt_tests,
            early_warning_model,
        ],
        jobs=[quarterly_refresh],
        schedules=[quarterly_schedule],
    )


def run_pipeline_simple() -> None:
    """Run the full pipeline without Dagster (for environments without it)."""
    from src.ingest import EdgarClient, ingest_bdc, create_sample_fixture, KNOWN_BDCS, get_user_agent

    logging.basicConfig(level=logging.INFO)

    logger.info("Step 1: Ingesting SEC filings")
    client = EdgarClient(get_user_agent())
    for bdc in KNOWN_BDCS:
        ingest_bdc(client, bdc, Path("data/raw"))
    create_sample_fixture(Path("data/raw"), Path("data/sample"))

    logger.info("Step 2: Loading to DuckDB")
    import duckdb
    db_path = Path("data/processed/warehouse.duckdb")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    all_records = []
    for bdc_dir in sorted(Path("data/raw").iterdir()):
        records_file = bdc_dir / "soi_records.json"
        if records_file.exists():
            records = json.loads(records_file.read_text())
            all_records.extend(records)
    if all_records:
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".json"))
        tmp.write_text(json.dumps(all_records))
        con.execute("DROP TABLE IF EXISTS raw_soi_positions")
        con.execute(f"CREATE TABLE raw_soi_positions AS SELECT * FROM read_json_auto('{tmp}')")
        tmp.unlink()
    con.close()

    logger.info("Step 3: Entity resolution")
    from src.entity_resolution import resolve_and_save
    resolve_and_save(db_path)

    logger.info("Step 4: dbt run")
    import subprocess
    subprocess.run(["dbt", "run", "--project-dir", "transforms", "--profiles-dir", "transforms"])
    subprocess.run(["dbt", "test", "--project-dir", "transforms", "--profiles-dir", "transforms"])

    logger.info("Pipeline complete")


if __name__ == "__main__":
    run_pipeline_simple()

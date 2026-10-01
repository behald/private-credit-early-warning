"""Run the full Private Credit Radar pipeline without dbt.

Steps:
0. Ingest real SEC BDC data (data/raw/soi_all_bdcs.parquet) if present,
   otherwise fall back to the synthetic sample
1. Load raw data into DuckDB
2. Create staging table
3. Run entity resolution
4. Create quarterly panel
5. Compute signals + watchlist
6. Train model

7. Run the rolling backtest and laggard alert (src/newly_distressed.py)

Usage:
    python run_pipeline.py            # real data if available, else sample
    python run_pipeline.py --sample   # force the synthetic sample
"""
import os, sys
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import duckdb

DB_PATH = Path("data/processed/warehouse.duckdb")


def load_sample_if_needed():
    """Load sample data into DuckDB if raw_soi_positions doesn't exist."""
    import tempfile

    sample = Path("data/sample/soi_sample.json")
    if not sample.exists():
        logger.error(f"No sample data at {sample}")
        sys.exit(1)

    con = duckdb.connect(str(DB_PATH))
    try:
        count = con.execute("SELECT COUNT(*) FROM raw_soi_positions").fetchone()[0]
        if count > 0:
            con.close()
            return
    except Exception:
        pass

    logger.info("Loading sample data into DuckDB...")
    records = json.loads(sample.read_text())
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(records))
    con.execute("DROP TABLE IF EXISTS raw_soi_positions")
    con.execute(f"CREATE TABLE raw_soi_positions AS SELECT * FROM read_json_auto('{tmp.as_posix()}')")
    tmp.unlink()
    count = con.execute("SELECT COUNT(*) FROM raw_soi_positions").fetchone()[0]
    logger.info(f"Loaded {count} sample records")
    con.close()


REAL_DATA = Path("data/raw/soi_all_bdcs.parquet")


def load_real_data() -> None:
    """Rebuild raw_soi_positions from the real SEC BDC data."""
    sys.path.insert(0, str(Path("src")))
    from ingest_sec_bdc import clean, load_source, summarize, write_warehouse
    logger.info(f"Ingesting real SEC BDC data from {REAL_DATA}")
    df = clean(load_source(REAL_DATA))
    write_warehouse(df, DB_PATH)
    summarize(df)


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    use_sample = "--sample" in sys.argv
    if not use_sample and REAL_DATA.exists():
        load_real_data()
    else:
        if not use_sample:
            logger.warning(f"{REAL_DATA} not found, using synthetic sample data")
        con = duckdb.connect(str(DB_PATH))
        con.execute("DROP TABLE IF EXISTS raw_soi_positions")
        con.close()
        load_sample_if_needed()

    con = duckdb.connect(str(DB_PATH))

    count = con.execute("SELECT COUNT(*) FROM raw_soi_positions").fetchone()[0]
    logger.info(f"Raw SOI positions: {count}")
    if count == 0:
        logger.error("No raw data.")
        sys.exit(1)

    logger.info("Creating staging table...")
    raw_cols = {r[0] for r in con.execute("DESCRIBE raw_soi_positions").fetchall()}
    # Real SEC data has asset_class: keep debt only so equity swings don't
    # pollute loan marks. Zero cost rows (undrawn revolvers) give infinite marks.
    extra_filter = "AND cost > 0"
    if "asset_class" in raw_cols:
        extra_filter += " AND asset_class = 'debt'"
    con.execute(f"""
        CREATE OR REPLACE TABLE stg_soi_positions AS
        SELECT
            cik,
            bdc_name,
            accession_number,
            CAST(filing_date AS DATE) AS filing_date,
            CAST(period_end AS DATE) AS period_end,
            fiscal_year,
            fiscal_period,
            frame,
            investment_id,
            borrower_name_raw,
            CAST(fair_value AS DOUBLE) AS fair_value,
            CAST(cost AS DOUBLE) AS cost,
            industry,
            investment_type,
            DATE_TRUNC('quarter', CAST(period_end AS DATE)) AS quarter
        FROM raw_soi_positions
        WHERE fair_value IS NOT NULL
            AND borrower_name_raw IS NOT NULL
            AND CAST(fair_value AS DOUBLE) > 0
            AND CAST(fair_value AS DOUBLE) / CAST(cost AS DOUBLE) BETWEEN 0.05 AND 2.0
            {extra_filter}
    """)
    pre_filter_sql = "SELECT COUNT(*) FROM raw_soi_positions WHERE fair_value IS NOT NULL AND borrower_name_raw IS NOT NULL AND cost > 0"
    if "asset_class" in raw_cols:
        pre_filter_sql += " AND asset_class = 'debt'"
    pre_filter = con.execute(pre_filter_sql).fetchone()[0]
    stg_count = con.execute("SELECT COUNT(*) FROM stg_soi_positions").fetchone()[0]
    dropped = pre_filter - stg_count
    zero_fv = con.execute(pre_filter_sql + " AND CAST(fair_value AS DOUBLE) = 0").fetchone()[0]
    neg_marks = con.execute(pre_filter_sql + " AND (CAST(fair_value AS DOUBLE) < 0 OR CAST(fair_value AS DOUBLE) / CAST(cost AS DOUBLE) < 0)").fetchone()[0]
    logger.info(f"Staging: {stg_count} rows (mark in [0.05, 2.0])")
    logger.info(f"  Dropped {dropped}: {zero_fv} fair_value=0, {neg_marks} negative marks (data errors), {dropped - zero_fv - neg_marks} other")

    logger.info("Running entity resolution...")
    con.close()

    sys.path.insert(0, str(Path("src")))
    from entity_resolution import resolve_and_save
    mapping = resolve_and_save(DB_PATH, use_embeddings=False)
    unique_canonical = len(set(mapping.values()))
    logger.info(f"Entity resolution: {len(mapping)} names -> {unique_canonical} canonical borrowers")

    con = duckdb.connect(str(DB_PATH))

    logger.info("Joining canonical IDs into staging...")
    con.execute("""
        CREATE OR REPLACE TABLE stg_soi_positions_resolved AS
        SELECT
            s.*,
            COALESCE(m.canonical_borrower_id, s.investment_id) AS canonical_borrower_id
        FROM stg_soi_positions s
        LEFT JOIN borrower_canonical_map m
            ON s.borrower_name_raw = m.borrower_name_raw
    """)

    con.execute("""
        CREATE OR REPLACE TABLE dim_borrower AS
        WITH raw AS (
            SELECT canonical_borrower_id,
                   REGEXP_REPLACE(
                       REGEXP_REPLACE(MODE(borrower_name_raw),
                           '^Issuer Name\s+', '', 'i'),
                       '^(?:[\w/.,-]+\s+)*?(?:Services?|Products?|Equipment|Solutions?|Industries?|Distribution)\s+',
                       '', 'i'
                   ) AS cleaned,
                   COUNT(DISTINCT cik) AS lenders_ever
            FROM stg_soi_positions_resolved
            GROUP BY canonical_borrower_id
        ),
        pass2 AS (
            SELECT canonical_borrower_id,
                   REGEXP_REPLACE(
                       REGEXP_REPLACE(cleaned, '^[&\s]+', ''),
                       '^(?:[\w/.,-]+\s+)*?(?:Services?|Products?|Equipment|Solutions?|Industries?|Distribution)\s+',
                       '', 'i'
                   ) AS borrower_name,
                   lenders_ever
            FROM raw
        )
        SELECT canonical_borrower_id,
               CASE
                   WHEN borrower_name IS NULL
                        OR TRIM(borrower_name) = ''
                        OR borrower_name ~ '^[0-9a-f]{6,}$'
                        OR LENGTH(TRIM(borrower_name)) <= 2
                   THEN 'Unnamed borrower'
                   ELSE TRIM(borrower_name)
               END AS borrower_name,
               lenders_ever
        FROM pass2
    """)

    # Written-off positions: fair_value=0 or mark in [0, 0.05). Negative marks dropped as data errors.
    wo_asset = "AND asset_class = 'debt'" if "asset_class" in raw_cols else ""
    con.execute(f"""
        CREATE OR REPLACE TABLE stg_written_off AS
        SELECT
            COALESCE(m.canonical_borrower_id, r.investment_id) AS canonical_borrower_id,
            r.borrower_name_raw,
            r.cik,
            r.bdc_name,
            DATE_TRUNC('quarter', CAST(r.period_end AS DATE)) AS quarter,
            CAST(r.fair_value AS DOUBLE) AS fair_value,
            CAST(r.cost AS DOUBLE) AS cost,
            CAST(r.fair_value AS DOUBLE) / NULLIF(CAST(r.cost AS DOUBLE), 0) AS mark
        FROM raw_soi_positions r
        LEFT JOIN borrower_canonical_map m ON r.borrower_name_raw = m.borrower_name_raw
        WHERE r.fair_value IS NOT NULL
            AND r.borrower_name_raw IS NOT NULL
            AND CAST(r.cost AS DOUBLE) > 0
            AND CAST(r.fair_value AS DOUBLE) >= 0
            AND CAST(r.fair_value AS DOUBLE) / CAST(r.cost AS DOUBLE) < 0.05
            {wo_asset}
    """)
    wo_count = con.execute("SELECT COUNT(*) FROM stg_written_off").fetchone()[0]
    logger.info(f"Written-off table: {wo_count} rows (fair_value=0 or mark < 0.05)")

    logger.info("Creating quarterly panel...")
    con.execute("""
        CREATE OR REPLACE TABLE mart_quarterly_panel AS
        SELECT
            canonical_borrower_id,
            cik AS lender_cik,
            bdc_name,
            quarter,
            SUM(fair_value) AS fair_value,
            SUM(cost) AS cost,
            MAX(CASE WHEN investment_type LIKE '%PIK%' THEN 1 ELSE 0 END) AS pik_flag,
            MAX(CASE WHEN investment_type LIKE '%NonAccrual%'
                      OR investment_type LIKE '%Non-Accrual%'
                      OR investment_type LIKE '%non_accrual%' THEN 1 ELSE 0 END) AS non_accrual_flag,
            MAX(industry) AS industry,
            COUNT(*) AS position_count
        FROM stg_soi_positions_resolved
        GROUP BY canonical_borrower_id, lender_cik, bdc_name, quarter
    """)
    panel_count = con.execute("SELECT COUNT(*) FROM mart_quarterly_panel").fetchone()[0]
    logger.info(f"Quarterly panel: {panel_count} rows")

    logger.info("Computing signals...")
    from signals import compute_signals, compute_watchlist
    compute_signals(con)
    compute_watchlist(con)

    signal_count = con.execute("SELECT COUNT(*) FROM mart_signals").fetchone()[0]
    watchlist_count = con.execute("SELECT COUNT(*) FROM mart_watchlist").fetchone()[0]
    logger.info(f"Signals: {signal_count} rows, Watchlist: {watchlist_count} entries")

    logger.info("Training model...")
    from model import prepare_model_data, time_based_split, train_baseline, train_main_model

    model_df = prepare_model_data(con, mark_threshold=0.80)
    logger.info(f"Model data: {len(model_df)} rows, positive rate: {model_df['target_distress_next_q'].mean():.3f}")

    if len(model_df) < 20:
        logger.warning("Too few rows for model training. Skipping.")
        con.close()
        return

    quarters = sorted(model_df["quarter"].unique())
    if len(quarters) < 3:
        logger.warning("Need at least 3 quarters. Skipping model.")
        con.close()
        return

    train_end = quarters[-3]
    val_q = quarters[-2]
    test_q = quarters[-1]

    train_df, val_df, test_df = time_based_split(model_df, str(train_end), str(val_q), str(test_q))
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df) if test_df is not None else 0}")

    if len(val_df) < 5:
        logger.warning("Validation set too small. Skipping model.")
        con.close()
        return

    baseline = train_baseline(train_df, val_df)
    logger.info(f"Baseline: PR-AUC={baseline['pr_auc']}, P@50={baseline['precision_at_50']}")

    model_dir = Path("models")
    main_metrics = train_main_model(train_df, val_df, model_dir)
    logger.info(f"LightGBM: PR-AUC={main_metrics.get('pr_auc')}, P@50={main_metrics.get('precision_at_50')}")

    tables = con.execute("SHOW TABLES").fetchall()
    logger.info(f"Tables: {[t[0] for t in tables]}")

    con.close()

    if not use_sample:
        logger.info("Running rolling backtest and laggard alert...")
        from newly_distressed import main as run_backtest
        run_backtest()

    logger.info("Pipeline complete! Start the dashboard with: streamlit run src/app.py")


if __name__ == "__main__":
    main()

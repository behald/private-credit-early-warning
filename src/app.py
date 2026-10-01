"""Streamlit dashboard for Private Credit Early Warning Radar.

Views: Market Overview, Borrower Drilldown, Lender Comparison, Watchlist,
Laggard Alerts, Backtest Results, Architecture.
Supports ?company= parameter for customization.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path("data/processed")
DB_PATH = DATA_DIR / "warehouse.duckdb"
METRICS_PATH = Path("models/metrics.json")
BACKTEST_PATH = Path("models/newly_distressed_metrics.json")
ALERTS_PATH = Path("models/laggard_alerts_latest.csv")


def has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0] > 0


def name_expr(con: duckdb.DuckDBPyConnection, alias: str = "s") -> tuple[str, str]:
    """SQL pieces to show a readable borrower name when dim_borrower exists."""
    if has_table(con, "dim_borrower"):
        return (f"COALESCE(d.borrower_name, {alias}.canonical_borrower_id)",
                f"LEFT JOIN dim_borrower d ON d.canonical_borrower_id = {alias}.canonical_borrower_id")
    return f"{alias}.canonical_borrower_id", ""
DISCLAIMER = (
    "**Disclaimer:** This is an independent research project using public SEC filings. "
    "It is not affiliated with any company mentioned. Not investment advice."
)


@st.cache_resource
def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


def load_company_config() -> dict | None:
    company = st.query_params.get("company")
    if not company:
        return None
    profile_path = Path(f"company_profiles/{company}.yaml")
    if profile_path.exists():
        import yaml
        return yaml.safe_load(profile_path.read_text())
    return None


def market_overview(con: duckdb.DuckDBPyConnection) -> None:
    st.header("Market Overview")

    try:
        summary = con.execute("""
            SELECT
                quarter,
                COUNT(DISTINCT canonical_borrower_id) AS borrowers,
                COUNT(DISTINCT lender_cik) AS lenders,
                MEDIAN(mark) AS median_mark,
                SUM(CASE WHEN non_accrual_flag = 1 THEN 1 ELSE 0 END) AS non_accrual_count,
                SUM(CASE WHEN pik_flag = 1 THEN 1 ELSE 0 END) AS pik_count
            FROM mart_signals
            GROUP BY quarter
            ORDER BY quarter
        """).fetchdf()

        if summary.empty:
            st.info("No data available yet. Run the pipeline first.")
            return

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(summary, x="quarter", y="median_mark", title="Median Mark Over Time")
            fig.add_hline(y=0.95, line_dash="dash", line_color="orange", annotation_text="Par")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(summary, x="quarter", y=["non_accrual_count", "pik_count"],
                        title="Stress Indicators by Quarter", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Total Borrowers (latest)", int(summary.iloc[-1]["borrowers"]))
        st.metric("Total Lenders (latest)", int(summary.iloc[-1]["lenders"]))

    except Exception as e:
        st.error(f"Query failed: {e}. Make sure the pipeline has been run.")


def borrower_drilldown(con: duckdb.DuckDBPyConnection) -> None:
    st.header("Borrower Drilldown")
    st.caption("Search a borrower to see how each lender marked it over time. Try Pluralsight.")

    try:
        name_sql, join_sql = name_expr(con)
        query = st.text_input("Search borrower name", value="Pluralsight")
        borrowers = con.execute(f"""
            SELECT s.canonical_borrower_id, {name_sql} AS borrower_name,
                   COUNT(DISTINCT s.lender_cik) AS lenders
            FROM mart_signals s {join_sql}
            WHERE {name_sql} ILIKE ?
            GROUP BY 1, 2
            ORDER BY lenders DESC
            LIMIT 200
        """, [f"%{query}%"]).fetchdf()

        if borrowers.empty:
            st.info("No borrower matches that search.")
            return

        labels = [f"{r.borrower_name} ({r.lenders} lenders)" for r in borrowers.itertuples()]
        pick = st.selectbox("Select Borrower", range(len(labels)), format_func=lambda i: labels[i])
        selected = borrowers.iloc[pick]["canonical_borrower_id"]
        title = borrowers.iloc[pick]["borrower_name"]

        detail = con.execute("""
            SELECT quarter, bdc_name, mark, mark_drift, pik_flag,
                   num_lenders, mark_dispersion, fair_value, cost
            FROM mart_signals
            WHERE canonical_borrower_id = ?
            ORDER BY quarter, bdc_name
        """, [selected]).fetchdf()

        fig = px.line(detail, x="quarter", y="mark", color="bdc_name", markers=True,
                      title=f"Mark history (fair value / cost): {title}")
        fig.add_hline(y=0.95, line_dash="dash", line_color="orange")
        fig.add_hline(y=0.80, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Position Details")
        st.dataframe(detail, use_container_width=True)

    except Exception as e:
        st.error(f"Query failed: {e}")


def lender_comparison(con: duckdb.DuckDBPyConnection) -> None:
    st.header("Lender Comparison")

    try:
        lenders = con.execute("""
            SELECT
                bdc_name,
                COUNT(DISTINCT canonical_borrower_id) AS portfolio_size,
                MEDIAN(mark) AS median_mark,
                AVG(mark_drift) AS avg_mark_drift,
                SUM(CASE WHEN non_accrual_flag = 1 THEN 1 ELSE 0 END) AS non_accrual_count,
                SUM(CASE WHEN pik_flag = 1 THEN 1 ELSE 0 END) AS pik_count
            FROM mart_signals
            WHERE quarter = (SELECT MAX(quarter) FROM mart_signals)
            GROUP BY bdc_name
            HAVING COUNT(DISTINCT canonical_borrower_id) >= 20
            ORDER BY median_mark ASC
        """).fetchdf()

        if lenders.empty:
            st.info("No lender data available.")
            return

        bottom20 = lenders.head(20)
        top20 = lenders.tail(20)
        chart_df = pd.concat([bottom20, top20]).drop_duplicates()

        fig = px.bar(chart_df, x="bdc_name", y="median_mark",
                     title="Median Mark by Lender — 20 Lowest & 20 Highest (min 20 positions)")
        fig.add_hline(y=0.95, line_dash="dash", line_color="orange")
        fig.update_yaxes(range=[0.85, chart_df["median_mark"].max() * 1.02])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(lenders, use_container_width=True)

    except Exception as e:
        st.error(f"Query failed: {e}")


def watchlist_view(con: duckdb.DuckDBPyConnection) -> None:
    st.header("Early Warning Watchlist")

    try:
        name_sql, join_sql = name_expr(con)

        # Heading-like names and equity positions to exclude
        junk_patterns = [
            "ILIKE '%Unsecured Notes%'",
            "ILIKE '%Common Equity%'",
            "ILIKE '%Common Stock%'",
            "ILIKE '%Preferred Stock%'",
            "ILIKE '%Affiliate Investments%'",
            "ILIKE '%Non-Affiliate%'",
            "ILIKE '%Total%'",
            "ILIKE '%Subtotal%'",
        ]
        junk_filter = " AND ".join(f"NOT ({name_sql} {p})" for p in junk_patterns)

        # Active watchlist: worst mark >= 0.3 (not already written off)
        watchlist = con.execute(f"""
            SELECT {name_sql} AS borrower_name, s.*
            FROM mart_watchlist s {join_sql}
            WHERE s.quarter = (SELECT MAX(quarter) FROM mart_watchlist)
                AND s.worst_mark >= 0.3
                AND {junk_filter}
            ORDER BY s.avg_mark ASC, s.max_dispersion DESC
            LIMIT 50
        """).fetchdf()

        if watchlist.empty:
            st.info("No watchlist entries. This is a good sign (or the pipeline has not run yet).")
            return

        st.warning(f"{len(watchlist)} borrowers on the watchlist")
        st.dataframe(watchlist, use_container_width=True)

        # Written-off positions from the dedicated staging table
        if has_table(con, "stg_written_off"):
            name_wo, join_wo = name_expr(con, alias="w")
            written_off = con.execute(f"""
                SELECT {name_wo} AS borrower_name,
                       w.canonical_borrower_id,
                       COUNT(DISTINCT w.cik) AS num_lenders,
                       MIN(w.mark) AS worst_mark,
                       AVG(w.mark) AS avg_mark,
                       SUM(w.cost) AS total_cost
                FROM stg_written_off w
                {join_wo}
                WHERE w.quarter = (SELECT MAX(quarter) FROM stg_written_off)
                GROUP BY 1, 2
                ORDER BY worst_mark ASC
                LIMIT 30
            """).fetchdf()

            if not written_off.empty:
                with st.expander(f"Written off or near zero ({len(written_off)} borrowers)"):
                    st.dataframe(written_off, use_container_width=True)

    except Exception as e:
        st.error(f"Query failed: {e}")


def laggard_alerts_view(con: duckdb.DuckDBPyConnection) -> None:
    st.header("Laggard Alerts")

    alert = None
    chosen_pts = None
    own_drift_pts = None
    if BACKTEST_PATH.exists():
        results = json.loads(BACKTEST_PATH.read_text())
        alert = next((r["laggard_alert"] for r in results if "laggard_alert" in r), None)
        if alert:
            chosen_pts = alert.get("chosen_threshold")
            own_drift_pts = alert.get("own_drift_threshold")

    if chosen_pts is not None:
        cut_label = f"{int(chosen_pts * 100)}"
        drift_label = f"{int(own_drift_pts * 100)}" if own_drift_pts else "2"
        st.markdown(
            f"An alert fires when **another manager cut a borrower by {cut_label} points "
            f"this quarter and this lender's mark moved less than {drift_label} points**. "
            f"Threshold chosen on pre-2024 data, validated on 2024+."
        )
    else:
        st.info("Run the pipeline to compute alert thresholds.")

    if alert:
        st.subheader("Threshold selection (pre-2024, Pluralsight excluded)")
        st.caption(f"Base rate of a 5+ point markdown: {alert.get('base_rate_excl_pluralsight_train', 0):.1%}")
        st.dataframe(pd.DataFrame(alert["threshold_sweep_train"]), use_container_width=True)

        if alert.get("threshold_sweep_test"):
            st.subheader("Out-of-sample validation (2024+)")
            st.caption(f"Base rate: {alert.get('base_rate_excl_pluralsight_test', 0):.1%}")
            st.dataframe(pd.DataFrame(alert["threshold_sweep_test"]), use_container_width=True)

        st.info(f"Chosen threshold: others cut at least **{int(chosen_pts * 100)} points** (best hit rate on pre-2024 data)")

        if alert.get("pluralsight_alerts"):
            ps = alert["pluralsight_alerts"][0]
            st.success(
                f"Pluralsight, quarter starting {ps['quarter']}: alert fired for "
                f"{ps['alerts']} lenders, and {ps['hit_next_q']} cut 5+ points the next quarter."
            )

    st.subheader("Alerts in the latest quarter")
    if not ALERTS_PATH.exists():
        st.info("No alerts file yet. Run the pipeline first.")
        return
    alerts = pd.read_csv(ALERTS_PATH)
    if has_table(con, "dim_borrower"):
        names = con.execute("SELECT canonical_borrower_id, borrower_name FROM dim_borrower").fetchdf()
        alerts = alerts.merge(names, on="canonical_borrower_id", how="left")
        cols = ["borrower_name"] + [c for c in alerts.columns if c != "borrower_name"]
        alerts = alerts[cols]
    alerts = alerts.rename(columns={"others_min_drift": "biggest_cut_by_other_manager"})
    st.warning(f"{len(alerts)} lender positions flagged")
    st.dataframe(alerts.sort_values("biggest_cut_by_other_manager"), use_container_width=True)


def backtest_view(con: duckdb.DuckDBPyConnection) -> None:
    st.header("Backtest Results (rolling, out of sample)")
    if not BACKTEST_PATH.exists():
        st.info("Run the pipeline to generate backtest results.")
        return
    results = [r for r in json.loads(BACKTEST_PATH.read_text()) if "experiment" in r]
    st.markdown(
        "Each quarter, models train only on earlier quarters. Scores are PR-AUC "
        "(higher is better). **rule** = lowest mark first, **old** = original features, "
        "**new** = adds most aggressive lender features, **v2** = adds manager level features."
    )
    for r in results:
        st.subheader(r["experiment"])
        c = st.columns(4)
        for col, key in zip(c, ["s_rule", "s_old", "s_new", "s_v2"]):
            col.metric(key.replace("s_", ""), r["pooled_ap"].get(key, "n/a"))
        st.dataframe(pd.DataFrame(r["per_quarter"]), use_container_width=True)
        d = r["new_minus_rule"]
        st.caption(f"New model minus rule: {d[0]:+.3f} (90% CI {d[1]:+.3f} to {d[2]:+.3f})")
        st.markdown("**Pluralsight, percentile rank (1.0 = most at risk):**")
        st.dataframe(pd.DataFrame(r["pluralsight"]), use_container_width=True)


def architecture_view(con: duckdb.DuckDBPyConnection) -> None:
    st.header("System Architecture")

    st.markdown("""
```
SEC EDGAR ──▶ DuckDB Warehouse ──▶ ML + Alerts ──▶ Streamlit
(7 zips)      raw → staging →       LightGBM        7 tabs
187K rows     entity resolution →   rolling backtest
              signals (96K) →       laggard alert
              watchlist (4.8K)
```
    """)

    st.subheader("Pipeline Stages")
    stages = {
        "Ingestion": "Reads SEC DERA BDC Schedule of Investments (soi.tsv from 7 quarterly zips). "
                     "Cleans mislabeled columns, deduplicates amended filings, filters to debt positions, "
                     "drops data errors. 187K raw → 161K clean.",
        "Entity Resolution": "Normalizes borrower names (strip legal suffixes, expand abbreviations), "
                             "hashes to stable canonical IDs. 17K raw names → 13K canonical borrowers. "
                             "Pure normalization, no embeddings needed.",
        "Signal Computation": "Mark (fair_value/cost), mark drift, cross-lender dispersion, "
                              "PIK/non-accrual flags, lender mark vs consensus. "
                              "All features use only current and prior quarter data.",
        "LightGBM Model": "10 features, time-based split, deterministic mode. "
                          "PR-AUC 0.76 on validation (inflated — includes already-distressed).",
        "Rolling Backtest": "Honest evaluation on only healthy loans. "
                            "Target B (5+ pt markdown): PR-AUC 0.158, 5x random, 1.5x rule.",
        "Laggard Alert": "Fires when another manager cuts a borrower 4+ pts and this lender hasn't moved. "
                         "Cost stability filter prevents portfolio reshuffle false alarms. "
                         "3.4x lift out of sample.",
    }
    for name, desc in stages.items():
        st.markdown(f"**{name}:** {desc}")

    st.subheader("Model Evaluation (pipeline default target)")
    st.caption("This target includes loans that are already distressed, which inflates the score. "
               "See the Backtest Results tab for the honest newly distressed evaluation.")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        col1, col2, col3 = st.columns(3)
        col1.metric("PR-AUC", metrics.get("pr_auc", "N/A"))
        col2.metric("Precision@50", metrics.get("precision_at_50", "N/A"))
        col3.metric("Positive Rate", metrics.get("val_positive_rate", "N/A"))

        if "feature_importance" in metrics:
            st.subheader("Feature Importance")
            imp = metrics["feature_importance"]
            imp_df = pd.DataFrame(
                {"Feature": list(imp.keys()), "Importance": list(imp.values())}
            ).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                        title="LightGBM Feature Importance (Gain)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model metrics not yet available. Run the pipeline to generate them.")

    st.subheader("Data Source")
    try:
        stats = con.execute("""
            SELECT
                COUNT(*) AS total_positions,
                COUNT(DISTINCT bdc_name) AS lenders,
                COUNT(DISTINCT canonical_borrower_id) AS borrowers,
                MIN(quarter) AS first_quarter,
                MAX(quarter) AS last_quarter
            FROM mart_signals
        """).fetchdf()
        if not stats.empty:
            row = stats.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Positions", f"{int(row['total_positions']):,}")
            c2.metric("Lenders", int(row['lenders']))
            c3.metric("Borrowers", int(row['borrowers']))
            st.caption(f"Data range: {row['first_quarter']} to {row['last_quarter']}")
    except Exception:
        st.info("Run the pipeline to see data statistics.")

    st.markdown("---")
    st.caption("Full architecture documentation: [docs/ARCHITECTURE.md](https://github.com/behaldivaye/private-credit-radar/blob/main/docs/ARCHITECTURE.md)")


def main() -> None:
    st.set_page_config(page_title="Private Credit Radar", layout="wide")
    st.title("Private Credit Early Warning Radar")
    st.caption(DISCLAIMER)

    company_config = load_company_config()
    if company_config:
        st.info(f"Viewing through lens of: **{company_config.get('name', '')}**")

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}. Run the pipeline first: `make run`")
        return

    con = get_connection().cursor()

    views = {
        "Market Overview": market_overview,
        "Borrower Drilldown": borrower_drilldown,
        "Lender Comparison": lender_comparison,
        "Early Warning Watchlist": watchlist_view,
        "Laggard Alerts": laggard_alerts_view,
        "Backtest Results": backtest_view,
        "Architecture & Eval": architecture_view,
    }

    highlight = None
    if company_config:
        highlight = company_config.get("highlight_views", [])

    tab_names = list(views.keys())
    tabs = st.tabs(tab_names)

    for tab, (name, view_fn) in zip(tabs, views.items()):
        with tab:
            if highlight and name.lower().replace(" ", "_") in [h.lower() for h in highlight]:
                st.success("Highlighted for this company profile")
            view_fn(con)


if __name__ == "__main__":
    main()

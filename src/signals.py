"""Signal computation for early warning detection.

Computes borrower-level and lender-level signals from the quarterly panel.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def compute_mark(fair_value: float, cost: float) -> float | None:
    """Mark = fair_value / cost. Returns None if cost is zero or missing."""
    if cost is None or cost == 0:
        return None
    return fair_value / cost


def compute_signals(con: duckdb.DuckDBPyConnection) -> None:
    """Compute all signal columns in the quarterly panel.

    Assumes mart_quarterly_panel exists with:
    canonical_borrower_id, lender_cik, quarter, fair_value, cost, pik_flag, non_accrual_flag
    """

    # Mark drift: change in mark from prior quarter for same borrower-lender
    con.execute("""
        CREATE OR REPLACE TABLE mart_signals AS
        WITH panel AS (
            SELECT *,
                fair_value / NULLIF(cost, 0) AS mark,
                ROW_NUMBER() OVER (
                    PARTITION BY canonical_borrower_id, lender_cik
                    ORDER BY quarter
                ) AS quarter_seq
            FROM mart_quarterly_panel
        ),
        with_lag AS (
            SELECT p.*,
                LAG(p.mark) OVER (
                    PARTITION BY p.canonical_borrower_id, p.lender_cik
                    ORDER BY p.quarter
                ) AS prior_mark,
                LAG(p.pik_flag, 1, 0) OVER (
                    PARTITION BY p.canonical_borrower_id, p.lender_cik
                    ORDER BY p.quarter
                ) AS prior_pik_flag
            FROM panel p
        ),
        cross_lender AS (
            SELECT
                canonical_borrower_id,
                quarter,
                COUNT(DISTINCT lender_cik) AS num_lenders,
                AVG(mark) AS avg_mark_across_lenders,
                STDDEV(mark) AS mark_dispersion,
                MIN(mark) AS min_mark,
                MAX(mark) AS max_mark
            FROM panel
            WHERE mark IS NOT NULL
            GROUP BY canonical_borrower_id, quarter
        )
        SELECT
            wl.*,
            wl.mark - wl.prior_mark AS mark_drift,
            CASE WHEN wl.prior_pik_flag = 0 AND wl.pik_flag = 1 THEN 1 ELSE 0 END AS pik_migration,
            CASE WHEN wl.prior_mark >= 0.95 AND wl.mark < 0.95 THEN 1 ELSE 0 END AS par_migration,
            cl.num_lenders,
            cl.avg_mark_across_lenders,
            cl.mark_dispersion,
            cl.min_mark,
            cl.max_mark,
            wl.mark - cl.avg_mark_across_lenders AS lender_mark_vs_avg
        FROM with_lag wl
        LEFT JOIN cross_lender cl
            ON wl.canonical_borrower_id = cl.canonical_borrower_id
            AND wl.quarter = cl.quarter
    """)

    # Lender lag: which lender marks down last for deteriorating borrowers
    con.execute("""
        CREATE OR REPLACE TABLE mart_lender_lag AS
        WITH deteriorating AS (
            SELECT canonical_borrower_id, quarter
            FROM mart_signals
            WHERE mark_drift < -0.05
            GROUP BY canonical_borrower_id, quarter
            HAVING COUNT(DISTINCT lender_cik) >= 2
        ),
        lender_marks AS (
            SELECT
                s.canonical_borrower_id,
                s.lender_cik,
                s.bdc_name,
                s.quarter,
                s.mark,
                s.mark_drift,
                RANK() OVER (
                    PARTITION BY s.canonical_borrower_id, s.quarter
                    ORDER BY s.mark_drift ASC
                ) AS markdown_rank
            FROM mart_signals s
            INNER JOIN deteriorating d
                ON s.canonical_borrower_id = d.canonical_borrower_id
                AND s.quarter = d.quarter
        )
        SELECT *,
            CASE WHEN markdown_rank = 1 THEN 'first_to_mark_down'
                 ELSE 'lagging' END AS lender_speed
        FROM lender_marks
    """)

    logger.info("Signal computation complete")


def compute_watchlist(con: duckdb.DuckDBPyConnection, mark_threshold: float = 0.85) -> None:
    """Build the early warning watchlist from signals."""
    con.execute(f"""
        CREATE OR REPLACE TABLE mart_watchlist AS
        SELECT
            canonical_borrower_id,
            quarter,
            COUNT(DISTINCT lender_cik) AS num_lenders,
            AVG(mark) AS avg_mark,
            MIN(mark) AS worst_mark,
            MAX(mark_dispersion) AS max_dispersion,
            SUM(pik_migration) AS pik_migrations,
            SUM(par_migration) AS par_migrations,
            MAX(CASE WHEN non_accrual_flag = 1 THEN 1 ELSE 0 END) AS any_non_accrual,
            AVG(mark_drift) AS avg_mark_drift
        FROM mart_signals
        GROUP BY canonical_borrower_id, quarter
        HAVING avg_mark < {mark_threshold}
            OR max_dispersion > 0.1
            OR pik_migrations > 0
            OR any_non_accrual > 0
        ORDER BY avg_mark ASC
    """)

    logger.info("Watchlist built")

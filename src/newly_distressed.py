"""Newly distressed experiment.

Question: can cross lender signals predict which HEALTHY loans turn
distressed next quarter, better than simply sorting by mark?

Target (per borrower x lender x quarter):
    mark >= 0.80 today  AND  mark < 0.80 next quarter
Rows already below 0.80 are excluded, since predicting that a bad loan
stays bad is trivial and inflates metrics.

Rows where cost jumps or collapses next quarter (ratio outside 0.67 to 1.5)
are excluded as likely restructurings, repayments or refinancings, where
the next quarter mark is not comparable.

New features (the "most aggressive lender" idea):
    gap_to_most_aggressive  this lender's mark minus the lowest mark any lender holds
    min_mark                the most aggressive lender's mark
    min_mark_drift          the biggest single quarter markdown by any lender
    share_marking_down      share of lenders that cut the mark this quarter
    min_mark_change_2q      change in the lowest mark over two quarters
    mark_drift_2q           this lender's two quarter change
    spread, interest_rate   loan pricing, a proxy for risk at origination

Two targets are tested:
    A  crosses 0.80 next quarter
    B  drops 5 points or more next quarter (tests whether lagging lenders catch up)

Evaluation: rolling origin. For each test quarter T, train only on
quarters before T. Scored from each lender's point of view, against the
rule "lowest own mark first". Confidence intervals resample whole
borrowers, since one borrower can appear at many lenders.

Usage:
    python src/newly_distressed.py
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

DB_PATH = Path("data/processed/warehouse.duckdb")
OUT_PATH = Path("models/newly_distressed_metrics.json")
THRESHOLD = 0.80
BIG_DROP = 0.05

OLD_FEATURES = [
    "mark", "mark_drift", "pik_flag", "num_lenders", "mark_dispersion",
    "lender_mark_vs_avg", "pik_migration", "par_migration", "avg_mark_across_lenders",
]
NEW_FEATURES = [
    "gap_to_most_aggressive", "min_mark", "min_mark_drift", "share_marking_down",
    "min_mark_change_2q", "mark_drift_2q", "spread", "interest_rate",
]
# Version 2: cross lender features computed across MANAGERS, excluding
# the lender's own manager, plus how the gap changed this quarter.
V2_FEATURES = [
    "num_managers", "others_min_mark", "gap_to_others", "gap_change",
    "others_min_drift", "others_share_cut", "lag_pressure",
]

# Funds run by one manager share a valuation process, so they are one voice.
MANAGER_PATTERNS = [
    (r"BLUE OWL|OWL ROCK", "blue_owl"), (r"^ARES", "ares"), (r"GOLUB", "golub"),
    (r"GOLDMAN", "goldman"), (r"OAKTREE", "oaktree"), (r"BLACKROCK", "blackrock"),
    (r"BLACKSTONE", "blackstone"), (r"FRANKLIN BSP", "franklin_bsp"), (r"BARINGS", "barings"),
    (r"^HPS", "hps"), (r"FS KKR|KKR FS|FS SPECIALTY", "kkr_fs"),
    (r"MORGAN STANLEY|NORTH HAVEN|SL INVESTMENT|T SERIES|LGAM", "morgan_stanley"),
    (r"APOLLO|MIDCAP", "apollo"), (r"CARLYLE", "carlyle"), (r"NUVEEN", "nuveen_churchill"),
    (r"SIXTH STREET", "sixth_street"), (r"PENNANTPARK", "pennantpark"),
    (r"NEW MOUNTAIN|NMF", "new_mountain"), (r"TWIN BROOK", "twin_brook"),
    (r"OHA|T\. ROWE", "oha"), (r"MAIN STREET|MSC INCOME", "main_street"), (r"^SLR", "slr"),
    (r"^TCW", "tcw"), (r"CRESCENT", "crescent"), (r"^BAIN", "bain"), (r"^AB ", "ab"),
    (r"FIDELITY", "fidelity"), (r"STELLUS", "stellus"), (r"MONROE", "monroe"),
    (r"INVESTCORP", "investcorp"), (r"GLADSTONE", "gladstone"), (r"VENTURE LENDING|WTI FUND", "wti"),
]


def manager_of(bdc_name: str) -> str:
    name = (bdc_name or "").upper()
    for pattern, manager in MANAGER_PATTERNS:
        if re.search(pattern, name):
            return manager
    return name.split()[0].lower() if name else "unknown"


def add_manager_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross lender features using other managers only."""
    df["manager"] = df["bdc_name"].map(manager_of)
    m = (df.groupby(["canonical_borrower_id", "quarter", "manager"])
           .agg(m_mark=("mark", "mean"), m_drift=("mark_drift", "mean")).reset_index())
    m["m_drift"] = m["m_drift"].clip(lower=-0.30)
    m["m_cut"] = (m["m_drift"] < -0.01).astype(float)

    key = ["canonical_borrower_id", "quarter"]
    g = m.groupby(key)
    m["num_managers"] = g["manager"].transform("size")

    def others_min(col: str) -> pd.Series:
        # Min over the other managers: use the second smallest when this manager is the smallest
        ranked = m.sort_values(key + [col])
        first = ranked.groupby(key)[col].transform("first")
        second = ranked.groupby(key)[col].transform(lambda s: s.iloc[1] if len(s) > 1 else np.nan)
        is_min = ranked[col] == first
        return pd.Series(np.where(is_min, second, first), index=ranked.index).reindex(m.index)

    m["others_min_mark"] = others_min("m_mark")
    m["others_min_drift"] = others_min("m_drift")
    total_cut = g["m_cut"].transform("sum")
    m["others_share_cut"] = ((total_cut - m["m_cut"].fillna(0)) /
                             (m["num_managers"] - 1)).where(m["num_managers"] > 1)

    df = df.merge(m[key + ["manager", "num_managers", "others_min_mark",
                           "others_min_drift", "others_share_cut"]],
                  on=key + ["manager"], how="left")
    df["gap_to_others"] = df["mark"] - df["others_min_mark"]
    # How much the gap opened this quarter: positive when others just cut and I didn't
    df = df.sort_values(["canonical_borrower_id", "lender_cik", "quarter"])
    df["gap_change"] = df["gap_to_others"] - df.groupby(
        ["canonical_borrower_id", "lender_cik"])["gap_to_others"].shift(1)
    df["lag_pressure"] = df["mark_drift"] - df["others_min_drift"]
    return df


LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, seed=42, num_threads=1,
)
NUM_ROUNDS = 300


def build_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        WITH pricing AS (
            SELECT r.canonical_borrower_id, r.cik AS lender_cik, r.quarter,
                   AVG(raw.spread) AS spread, AVG(raw.interest_rate) AS interest_rate
            FROM stg_soi_positions_resolved r
            JOIN raw_soi_positions raw
              ON raw.investment_id = r.investment_id
             AND CAST(raw.period_end AS DATE) = r.period_end
            GROUP BY 1, 2, 3
        )
        SELECT s.*, p.spread, p.interest_rate
        FROM mart_signals s
        LEFT JOIN pricing p USING (canonical_borrower_id, lender_cik, quarter)
    """).fetchdf()

    df["quarter"] = pd.to_datetime(df["quarter"])
    df = df.sort_values(["canonical_borrower_id", "lender_cik", "quarter"])
    g = df.groupby(["canonical_borrower_id", "lender_cik"])

    # Next quarter outcome, only when the next row really is the next quarter
    df["next_quarter"] = g["quarter"].shift(-1)
    df["next_mark"] = g["mark"].shift(-1)
    df["next_cost"] = g["cost"].shift(-1)
    consecutive = (df["next_quarter"] - df["quarter"]).dt.days.between(80, 100)
    cost_ratio = df["next_cost"] / df["cost"]
    comparable = consecutive & cost_ratio.between(0.67, 1.5)

    # Two quarter history for this lender
    df["mark_drift_2q"] = df["mark"] - g["mark"].shift(2)

    # Borrower level cross lender features
    b = df.groupby(["canonical_borrower_id", "quarter"])
    df["min_mark_drift"] = b["mark_drift"].transform("min")
    df["share_marking_down"] = b["mark_drift"].transform(lambda s: (s < -0.01).mean())
    df["gap_to_most_aggressive"] = df["mark"] - df["min_mark"]

    bq = (df.groupby(["canonical_borrower_id", "quarter"])["min_mark"].first()
            .reset_index().sort_values(["canonical_borrower_id", "quarter"]))
    bq["min_mark_change_2q"] = bq["min_mark"] - bq.groupby("canonical_borrower_id")["min_mark"].shift(2)
    df = df.merge(bq[["canonical_borrower_id", "quarter", "min_mark_change_2q"]],
                  on=["canonical_borrower_id", "quarter"], how="left")

    df = add_manager_features(df)
    healthy = df["mark"] >= THRESHOLD
    return df[healthy & comparable & df["next_mark"].notna()].copy()


def cluster_bootstrap_diff(frame: pd.DataFrame, a: str, b: str, n: int = 500, seed: int = 0):
    """AP(a) minus AP(b), resampling whole borrowers so a borrower held by
    many lenders can't dominate. Returns point estimate and 90% interval."""
    rng = np.random.default_rng(seed)
    y, sa, sb = frame["target"].to_numpy(), frame[a].to_numpy(), frame[b].to_numpy()
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(frame["canonical_borrower_id"].to_numpy()):
        groups.setdefault(g, []).append(i)
    keys = np.array(list(groups))
    point = average_precision_score(y, sa) - average_precision_score(y, sb)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        if y[idx].sum() > 0:
            diffs.append(average_precision_score(y[idx], sa[idx]) - average_precision_score(y[idx], sb[idx]))
    lo, hi = np.percentile(diffs, [5, 95])
    return round(point, 3), round(lo, 3), round(hi, 3)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    model = lgb.train(LGB_PARAMS, lgb.Dataset(train[features], label=train["target"]),
                      num_boost_round=NUM_ROUNDS)
    return model.predict(test[features]), model


def run_experiment(df: pd.DataFrame, name: str, pluralsight_ids: set[str]) -> dict:
    """Rolling origin evaluation from each lender's point of view."""
    quarters = sorted(df["quarter"].unique())
    per_q, pooled, psight = [], [], []
    model = None
    for T in quarters[2:]:
        train, test = df[df["quarter"] < T], df[df["quarter"] == T].copy()
        if test["target"].sum() == 0 or train["target"].sum() < 20:
            continue
        test["s_rule"] = -test["mark"]                      # lowest own mark first
        test["s_old"], _ = fit_predict(train, test, OLD_FEATURES)
        test["s_new"], _ = fit_predict(train, test, OLD_FEATURES + NEW_FEATURES)
        test["s_v2"], model = fit_predict(train, test, OLD_FEATURES + NEW_FEATURES + V2_FEATURES)
        multi = test[test["num_lenders"] >= 2]
        row = {"test_quarter": str(pd.Timestamp(T).date()), "rows": len(test),
               "positives": int(test["target"].sum()), "base_rate": round(test["target"].mean(), 4)}
        for col, label in [("s_rule", "rule"), ("s_old", "old"), ("s_new", "new"), ("s_v2", "v2")]:
            row[f"ap_{label}"] = round(average_precision_score(test["target"], test[col]), 3)
            if multi["target"].sum() > 0:
                row[f"ap_{label}_multi"] = round(average_precision_score(multi["target"], multi[col]), 3)
        per_q.append(row)
        pooled.append(test)

    allq = pd.concat(pooled, ignore_index=True)
    cols = ["s_rule", "s_old", "s_new", "s_v2"]
    for col in cols:
        allq[col + "_pct"] = allq.groupby("quarter")[col].rank(pct=True)
    pooled_ap = {c: round(average_precision_score(allq["target"], allq[c + "_pct"]), 3) for c in cols}

    # Manager level view: one row per manager x borrower x quarter, so the
    # five Blue Owl funds count once
    mgr = (allq.groupby(["canonical_borrower_id", "quarter", "manager"])
               .agg(target=("target", "max"), **{c: (c + "_pct", "mean") for c in cols})
               .reset_index())
    manager_ap = {c: round(average_precision_score(mgr["target"], mgr[c]), 3) for c in cols}

    ps = allq[allq["canonical_borrower_id"].isin(pluralsight_ids)]
    for q, g in ps.groupby("quarter"):
        psight.append({
            "quarter": str(pd.Timestamp(q).date()), "lenders": len(g),
            "lenders_hit_next_q": int(g["target"].sum()),
            "median_pct_rank_new": round(g["s_new_pct"].median(), 3),
            "median_pct_rank_v2": round(g["s_v2_pct"].median(), 3),
            "median_pct_rank_rule": round(g["s_rule_pct"].median(), 3),
        })

    top = sorted(zip(model.feature_name(), model.feature_importance("gain")), key=lambda kv: -kv[1])[:10]
    return {
        "experiment": name,
        "per_quarter": per_q,
        "pooled_ap": pooled_ap,
        "manager_level_ap": manager_ap,
        "new_minus_rule": cluster_bootstrap_diff(allq, "s_new_pct", "s_rule_pct"),
        "new_minus_old": cluster_bootstrap_diff(allq, "s_new_pct", "s_old_pct"),
        "v2_minus_new": cluster_bootstrap_diff(allq, "s_v2_pct", "s_new_pct"),
        "v2_minus_rule": cluster_bootstrap_diff(allq, "s_v2_pct", "s_rule_pct"),
        "pluralsight": psight,
        "top_features": {k: round(float(v)) for k, v in top},
    }


def print_result(r: dict) -> None:
    pd.set_option("display.width", 220)
    print(f"\n=== {r['experiment']} ===")
    print(pd.DataFrame(r["per_quarter"]).to_string(index=False))
    print(f"Pooled AP: {r['pooled_ap']}")
    print(f"Manager level AP: {r['manager_level_ap']}")
    print(f"v2 minus v1:    {r['v2_minus_new'][0]:+.3f} (90% CI {r['v2_minus_new'][1]:+.3f} to {r['v2_minus_new'][2]:+.3f})")
    print(f"v2 minus rule:  {r['v2_minus_rule'][0]:+.3f} (90% CI {r['v2_minus_rule'][1]:+.3f} to {r['v2_minus_rule'][2]:+.3f})")
    print(f"New minus rule: {r['new_minus_rule'][0]:+.3f} (90% CI {r['new_minus_rule'][1]:+.3f} to {r['new_minus_rule'][2]:+.3f})")
    print(f"New minus old:  {r['new_minus_old'][0]:+.3f} (90% CI {r['new_minus_old'][1]:+.3f} to {r['new_minus_old'][2]:+.3f})")
    print("Pluralsight (percentile rank, 1.0 = most at risk):")
    print(pd.DataFrame(r["pluralsight"]).to_string(index=False))
    print("Top features:", ", ".join(r["top_features"]))


ALERT_OWN_DRIFT = -0.02     # "I have not moved": my mark fell less than 2 points
ALERT_THRESHOLDS = [-0.02, -0.03, -0.04, -0.05, -0.07, -0.10]
MIN_ALERTS_FOR_THRESHOLD = 100


def _sweep_thresholds(data: pd.DataFrame, base_rate: float) -> list[dict]:
    rows = []
    for th in ALERT_THRESHOLDS:
        a = data[(data["others_min_drift"] <= th) & (data["mark_drift"] > ALERT_OWN_DRIFT)]
        if len(a) == 0:
            rows.append({"others_cut_at_least": -th, "alerts": 0,
                         "hit_rate": 0.0, "lift": 0.0, "worst_quarter_hit": 0.0})
            continue
        hit = a["y"].mean()
        lift = hit / base_rate if base_rate > 0 else 0
        per_q = a.groupby("quarter")["y"].mean()
        rows.append({"others_cut_at_least": -th, "alerts": len(a),
                      "hit_rate": round(hit, 3), "lift": round(lift, 1),
                      "worst_quarter_hit": round(per_q.min(), 3)})
    return rows


def laggard_alert_report(df: pd.DataFrame, pluralsight_ids: set[str]) -> dict:
    """A simple, explainable alert: another manager just cut this borrower
    hard, and this lender has not moved. Evaluated on 5+ point markdowns.
    Threshold chosen on pre-2024 data, validated on 2024+."""
    df = df.copy()
    df["y"] = ((df["next_mark"] - df["mark"]) <= -BIG_DROP).astype(int)
    rest = df[~df["canonical_borrower_id"].isin(pluralsight_ids)]

    train = rest[rest["quarter"].dt.year < 2024]
    test = rest[rest["quarter"].dt.year >= 2024]
    base_train = train["y"].mean()
    base_test = test["y"].mean() if len(test) > 0 else 0

    sweep_train = _sweep_thresholds(train, base_train)
    sweep_test = _sweep_thresholds(test, base_test)

    best_th = ALERT_THRESHOLDS[0]
    best_hit = 0.0
    for th, row in zip(ALERT_THRESHOLDS, sweep_train):
        if row["alerts"] >= MIN_ALERTS_FOR_THRESHOLD and row["hit_rate"] > best_hit:
            best_hit = row["hit_rate"]
            best_th = th

    chosen = best_th
    logger.info(f"Laggard threshold chosen on pre-2024 data: {-chosen} points (hit rate {best_hit:.1%})")

    df["alert"] = (df["others_min_drift"] <= chosen) & (df["mark_drift"] > ALERT_OWN_DRIFT)
    ps = df[df["canonical_borrower_id"].isin(pluralsight_ids) & df["alert"]]
    ps_summary = [{"quarter": str(pd.Timestamp(q).date()), "alerts": len(g),
                   "hit_next_q": int(g["y"].sum())} for q, g in ps.groupby("quarter")]

    latest = df[df["quarter"] == df["quarter"].max()]
    return {"base_rate_excl_pluralsight_train": round(base_train, 4),
            "base_rate_excl_pluralsight_test": round(base_test, 4),
            "threshold_sweep_train": sweep_train,
            "threshold_sweep_test": sweep_test,
            "chosen_threshold": -chosen,
            "pluralsight_alerts": ps_summary,
            "latest_quarter_alerts": latest[latest["alert"]][
                ["canonical_borrower_id", "bdc_name", "manager", "mark", "others_min_drift"]]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    base = build_dataset(con)
    pluralsight_ids = {r[0] for r in con.execute(
        "SELECT DISTINCT canonical_borrower_id FROM borrower_canonical_map "
        "WHERE borrower_name_raw ILIKE '%pluralsight%'").fetchall()}
    con.close()

    # Experiment A: crosses the 0.80 line next quarter
    a = base.copy()
    a["target"] = (a["next_mark"] < THRESHOLD).astype(int)

    # Experiment B: big markdown next quarter (5 points or more).
    # Not tied to how close the mark already is to a fixed line,
    # so it tests whether lagging lenders catch up.
    b = base.copy()
    b["target"] = ((b["next_mark"] - b["mark"]) <= -BIG_DROP).astype(int)

    results = [
        run_experiment(a, "A: newly distressed (crosses 0.80)", pluralsight_ids),
        run_experiment(b, f"B: big markdown next quarter (>= {BIG_DROP:.2f})", pluralsight_ids),
    ]
    for r in results:
        print_result(r)

    alert = laggard_alert_report(base, pluralsight_ids)
    print("\n=== Laggard alert: another manager just cut, this lender has not ===")
    print(f"Base rate (pre-2024, Pluralsight excluded): {alert['base_rate_excl_pluralsight_train']}")
    print(f"Base rate (2024+, Pluralsight excluded):    {alert['base_rate_excl_pluralsight_test']}")
    print("Threshold sweep (pre-2024, used to choose):")
    print(pd.DataFrame(alert["threshold_sweep_train"]).to_string(index=False))
    print("Threshold sweep (2024+, out of sample):")
    print(pd.DataFrame(alert["threshold_sweep_test"]).to_string(index=False))
    print(f"Chosen: others cut at least {alert['chosen_threshold']:.2f}")
    print("Pluralsight alerts:", alert["pluralsight_alerts"])
    latest = alert.pop("latest_quarter_alerts")
    latest.to_csv(OUT_PATH.with_name("laggard_alerts_latest.csv"), index=False)
    results.append({"laggard_alert": alert})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

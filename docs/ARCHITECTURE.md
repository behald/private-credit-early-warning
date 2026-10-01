# Private Credit Early Warning Radar -- Architecture

**Divaye Behal | Independent Research | 2024**

An end-to-end pipeline that reads every loan US Business Development Companies report to the SEC each quarter, matches the same borrower across lenders, and flags loans likely to be marked down next quarter. Everything runs locally in Python and DuckDB -- no cloud services, no API keys, no cost.

| Stat | Value |
|---|---|
| Raw positions ingested | 187,301 |
| Clean debt positions | 161,017 |
| Unique borrowers (after entity resolution) | 13,251 |
| Lenders (BDCs) | 139, grouped into ~30 managers |
| Quarters covered | Q4 2022 -- Q3 2024 |
| Pipeline runtime (local, CPU) | ~45 seconds |

---

## 1. Architecture Diagram

```
 SEC EDGAR                DuckDB Warehouse                  ML + Alerts              Dashboard
 ─────────               ──────────────────                 ───────────              ─────────

 ┌───────────┐     ┌──────────────────────────────────┐    ┌────────────────┐    ┌─────────────┐
 │ soi.tsv   │     │                                  │    │                │    │             │
 │ (7 zips,  │────▶│  raw_soi_positions                │    │  LightGBM      │    │  Streamlit  │
 │  730K rows)│     │       │                          │    │  (10 features, │    │  7 tabs:    │
 └───────────┘     │       ▼                          │    │   time-split)  │    │  overview,  │
                   │  stg_soi_positions (161K)         │    │       │        │    │  drilldown, │
                   │       │                          │    │       ▼        │    │  lender,    │
 ┌───────────┐     │       ▼                          │    │  Predictions   │───▶│  watchlist, │
 │ Entity    │────▶│  stg_soi_positions_resolved       │    │  PR-AUC 0.76   │    │  alerts,   │
 │ Resolution│     │       │                          │    │  P@50 = 0.94   │    │  backtest,  │
 │ (MD5 +    │     │       ├──▶ mart_quarterly_panel   │    │                │    │  arch       │
 │  normalize)│     │       │                          │    └────────────────┘    │             │
 └───────────┘     │       ├──▶ mart_signals (96K)     │                         │             │
                   │       │                          │    ┌────────────────┐    │             │
                   │       ├──▶ mart_watchlist (4.8K)   │    │ Rolling        │    │             │
                   │       │                          │    │ Backtest       │───▶│             │
                   │       ├──▶ dim_borrower (13K)     │    │ (5 quarters,   │    │             │
                   │       │                          │    │  2 targets)    │    │             │
                   │       └──▶ stg_written_off        │    │       │        │    │             │
                   │                                  │    │       ▼        │    │             │
                   └──────────────────────────────────┘    │ Laggard Alert  │───▶│             │
                                                          │ (cost-stable   │    │             │
                                                          │  drift check)  │    └─────────────┘
                                                          └────────────────┘
```

Data flows left to right: SEC filings enter DuckDB, entity resolution links borrowers across lenders, signals feed both the ML model and the rule-based laggard alert, and everything surfaces in Streamlit.

---

## 2. Pipeline Steps

The pipeline runs as a single command (`python run_pipeline.py`) and executes these stages in order:

### Stage 0: Ingestion (`src/ingest_sec_bdc.py`)

Reads the SEC DERA BDC Schedule of Investments file (`soi.tsv` from 7 quarterly zips) and writes `raw_soi_positions` into DuckDB.

**What it cleans:**
- Fair value and cost sit under mislabeled columns in the SEC file -- remapped.
- Each filing repeats prior periods -- only each filing's own period is kept.
- Amended filings (multiple accession numbers for the same CIK + period) -- last filing wins.
- Subtotal, cash, and total rows removed by pattern and by size.
- Equity positions filtered out (`asset_class = 'debt'` only).
- Zero-cost rows (undrawn revolvers) dropped.
- Rows with `mark` (fair_value / cost) outside [0.05, 2.0] dropped as data errors. Positions with mark below 0.05 go to a separate `stg_written_off` table.

**Result:** 187,301 raw rows become 161,017 clean debt positions.

### Stage 1: Entity Resolution (`src/entity_resolution.py`)

Matches the same borrower across different lenders. BDCs each type borrower names slightly differently ("Acrisure LLC" vs "ACRISURE, LLC" vs "Acrisure Holdings").

**Method:**
1. Normalize: lowercase, strip legal suffixes (LLC, Inc, Corp...), expand abbreviations, remove punctuation.
2. Hash: MD5 of normalized name gives a stable `canonical_borrower_id`.
3. Exact match on normalized form groups most variants.
4. (Optional) Embedding similarity with blocking for remaining fuzzy matches.

The pipeline runs with `use_embeddings=False` -- pure normalization + hashing. This is deterministic and fast.

**Result:** 17,464 unique raw names resolve to 13,251 canonical borrowers. The `dim_borrower` table maps IDs to cleaned display names, with a two-pass regex to strip category prefixes ("Internet Software and Services Acrisure" becomes "Acrisure").

### Stage 2: Quarterly Panel (`run_pipeline.py`)

Aggregates positions to one row per borrower-lender-quarter in `mart_quarterly_panel`. Sums fair value and cost, flags PIK and non-accrual positions, counts positions per tranche.

### Stage 3: Signal Computation (`src/signals.py`)

Computes `mart_signals` with these features per position:

| Feature | What it measures |
|---|---|
| `mark` | fair_value / cost -- the core health signal |
| `mark_drift` | change in mark vs prior quarter |
| `prior_mark` | last quarter's mark (via LAG) |
| `num_lenders` | how many BDCs hold this borrower |
| `mark_dispersion` | std dev of marks across lenders for same borrower |
| `avg_mark_across_lenders` | consensus mark across all lenders |
| `lender_mark_vs_avg` | this lender's mark minus the consensus |
| `pik_migration` | newly flagged PIK this quarter |
| `par_migration` | newly returned to par this quarter |
| `pik_flag`, `non_accrual_flag` | binary status flags |

Also computes `mart_watchlist`: borrowers where any lender's mark is below 0.90 or drifted more than 5 points.

### Stage 4: LightGBM Model (`src/model.py`)

Trains a gradient-boosted model to predict: will this position become non-accrual or fall below mark 0.80 next quarter?

**Setup:**
- Time-based split: train on all quarters up to Q-2, validate on Q-1, test on Q.
- 10 features (the signal columns above).
- Class-weighted for imbalanced target (~3.7% positive rate).
- `deterministic=True, force_row_wise=True, num_threads=1, seed=42` for reproducibility.

**Result:** PR-AUC = 0.7579, Precision@50 = 0.94 on validation. This model includes already-distressed loans in the target, so the score is inflated. The honest evaluation is in the rolling backtest.

### Stage 5: Rolling Backtest (`src/newly_distressed.py`)

The real evaluation. Trains and predicts on a rolling quarterly basis, scoring only loans that are currently healthy (mark >= 0.80).

**Two targets:**
- **Target A: crosses below 0.80 next quarter.** Simple rule (sort by lowest mark) wins at PR-AUC 0.227. The model reaches 0.209. This target is almost mechanical -- if your mark is 0.82, you're likely to cross 0.80.
- **Target B: marked down 5+ points next quarter** (~3% base rate). The model wins at PR-AUC 0.158, about 5x better than random and 1.5x better than the simple rule. Cross-lender features add +0.012 PR-AUC (90% CI: +0.004 to +0.021).

### Stage 6: Laggard Alert (`src/newly_distressed.py`)

A rule-based alert that fires when another manager has cut a borrower's mark by 4+ points this quarter but this lender's mark moved less than 2 points.

**Cost stability filter:** Only considers cuts from managers whose total cost on the borrower stayed between 0.67x and 1.5x of the prior quarter. This prevents alerts from portfolio reshuffles being mistaken for credit calls.

**Threshold selection:** Swept on pre-2024 data (Pluralsight excluded), validated on 2024+.

| | Pre-2024 (train) | 2024+ (out of sample) |
|---|---|---|
| Alerts at 4 pts | 235 | 249 |
| Hit rate | 13.2% | 10.0% |
| Lift vs base rate | 4.8x | 3.4x |

**Pluralsight case study:** 14 alerts fired in Q3 2023 (Ares cut first, others hadn't moved). 10 of those 14 lenders were marked down 5+ points the next quarter.

---

## 3. Tools

| Tool | Role | Why this one |
|---|---|---|
| **Python 3.11** | Pipeline, model, dashboard | Standard for data/ML work |
| **DuckDB** | Analytical warehouse | Embedded, zero config, SQL on local files, fast aggregation |
| **pandas** | DataFrames | Interop between DuckDB and scikit-learn/LightGBM |
| **LightGBM** | Gradient boosting | Fast, handles imbalanced classes, deterministic mode |
| **scikit-learn** | Baseline model, metrics | LogisticRegression baseline, PR-AUC, precision@k |
| **Streamlit** | Dashboard | Interactive, Python-native, free hosting on Streamlit Cloud |
| **Plotly** | Charts | Interactive plots in Streamlit without JavaScript |
| **pydantic** | Data validation | Type-safe entity resolution data models |
| **pytest** | Tests | Standard Python testing |

**Not used (and why not):**
- **dbt:** Originally planned but dropped. DuckDB SQL in `run_pipeline.py` is simpler for a single-file warehouse with no incremental loads.
- **sentence-transformers:** Available in entity resolution but disabled (`use_embeddings=False`). Pure normalization + MD5 hashing gives good enough matching without a 200MB model download.
- **Airflow / Prefect:** No scheduler needed. The pipeline is a single script that runs in 45 seconds.

---

## 4. Tradeoffs

### Mark = fair_value / cost as the core signal

**Chose:** Ratio of fair value to cost as the universal health metric.
**Alternative:** Raw fair value, or spread-to-benchmark.
**Why:** Mark is what BDCs actually report and what their NAV depends on. It is comparable across positions of different sizes. Spread data is not in the SOI filing.
**Risk:** Cost can change (restructuring, add-on draws), making the ratio noisy. The cost stability filter mitigates this for the laggard alert.

### Simple rule vs ML for Target A

**Chose:** Report both, but acknowledge the rule wins.
**Alternative:** Force the model, add features until it beats the rule.
**Why:** Target A (crosses 0.80) is nearly deterministic from the current mark. Building a complex model to beat a sort would be overfitting to noise. The honest finding is that simple rules work for threshold-crossing and ML adds value for the harder "big markdown" target.

### Entity resolution without embeddings

**Chose:** Normalization + MD5 hashing only.
**Alternative:** Sentence-transformer embeddings with cosine similarity.
**Why:** The pipeline runs with `use_embeddings=False` because most borrower name variation is legal suffix and punctuation noise that normalization handles perfectly. Embeddings would add a 200MB model dependency and 30 seconds of runtime for marginal gains. The embedding path exists in the code and can be enabled.

### Laggard alert: cost-stable drift only

**Chose:** Only fire alerts when the cutting manager's cost position is stable (0.67x to 1.5x of prior quarter).
**Alternative:** Fire on any drift regardless of cost changes.
**Why:** Managers who sell down a position (cost drops) may also lower the mark to reflect the exit price, not a credit view. Requiring cost stability ensures the mark cut reflects a genuine credit reassessment. The threshold went from 7pts to 4pts after adding this filter because the signal got cleaner.

### Manager-level aggregation

**Chose:** Group BDCs into ~30 managers using regex patterns (e.g., all Owl Rock entities map to "Blue Owl").
**Alternative:** Treat each BDC CIK independently.
**Why:** A single manager filing under 4 CIKs (common for Ares, Blue Owl, Golub) would look like 4 independent lenders agreeing, when it is one entity's view. Manager grouping prevents false consensus signals.

### 7 quarters of data

**Chose:** Use all available SEC DERA BDC data (Q4 2022 -- Q3 2024).
**Alternative:** Wait for more history before publishing.
**Why:** The laggard alert and rolling backtest both use out-of-sample validation. Results are honest about the short history. The infrastructure is ready to ingest new quarters as the SEC publishes them.

---

## 5. Production Improvements

These would matter if this moved beyond a research project:

1. **Incremental ingestion.** Currently drops and rebuilds every table. A production system would detect new filings by accession number and append only new quarters, reducing runtime from 45 seconds to under 5.

2. **Entity resolution with embeddings + human review.** Enable the sentence-transformer path for fuzzy matches, surface uncertain pairs for manual labeling, and retrain the matching threshold. The code already supports a manual override CSV.

3. **Non-accrual extraction from footnotes.** Non-accrual status is buried in filing footnotes, not tagged columns. An NLP extraction pipeline (regex or LLM) could pull this from the raw XBRL, adding a strong distress signal.

4. **Alert delivery.** Push laggard alerts to Slack or email instead of requiring a dashboard visit. Include the borrower name, which managers cut, by how much, and the lender's current mark.

5. **Backtesting on longer history.** As the SEC publishes more quarters, the rolling backtest window grows and confidence intervals tighten. The pipeline handles any number of quarters without code changes.

6. **Scheduled pipeline.** Run on a cron or Airflow DAG after each quarterly SEC filing deadline. The pipeline is already idempotent -- safe to re-run.

7. **Model monitoring.** Track PR-AUC and alert hit rate per quarter. Alert if the model's lift drops below 2x or the laggard alert's hit rate drops below baseline for two consecutive quarters.

8. **Written-off integration.** The `stg_written_off` table captures positions with mark below 0.05 or fair value of zero. A production system would track the path from early warning to write-off, measuring the full lifecycle.

---

## 6. Results

All results are out-of-sample. Each quarter's model trains only on earlier quarters.

### LightGBM Rolling Backtest

**Target A -- Newly Distressed (crosses below 0.80)**

| Method | PR-AUC | Note |
|---|---|---|
| Rule: sort by lowest mark | **0.227** | Wins -- target is nearly mechanical |
| LightGBM, original features | 0.197 | |
| LightGBM, + cross-lender features | 0.209 | +0.012 vs original |

**Target B -- Big Markdown (5+ point drop next quarter, ~3% base rate)**

| Method | PR-AUC | Note |
|---|---|---|
| Rule: sort by lowest mark | 0.105 | |
| LightGBM, original features | 0.146 | |
| LightGBM, + cross-lender features | **0.158** | 5x random, 1.5x rule |

Cross-lender features add +0.012 PR-AUC (90% CI: +0.004 to +0.021, resampled by borrower).

### Laggard Alert

| Metric | Pre-2024 (train) | 2024+ (OOS) |
|---|---|---|
| Threshold | 4 pts | 4 pts |
| Alerts fired | 235 | 249 |
| Hit rate (marked down 5+ pts next Q) | 13.2% | 10.0% |
| Lift vs 3% base rate | 4.8x | 3.4x |

### Pluralsight End-to-End

| Quarter | Ares mark | Blue Owl mark | Golub mark | Alert status |
|---|---|---|---|---|
| Q3 2023 | 0.94 | 0.97 | 1.00 | 14 alerts fire (Ares cutting, others flat) |
| Q4 2023 | 0.89 | 0.97 | 1.00 | Ares deepens cut |
| Q1 2024 | 0.85 | 0.84 | 0.98 | 10 of 14 alerted lenders now marked down |
| Q2 2024 | 0.48 | 0.47 | 0.54 | All 15 lenders below 0.55 |

The rolling backtest model ranked Pluralsight lenders in the 97th percentile for risk the quarter before the collapse.

### Top Features (LightGBM Gain, Target B)

| Feature | Importance |
|---|---|
| avg_mark_across_lenders | 13,468 |
| min_mark (cross-lender) | 9,934 |
| mark | 9,217 |
| spread | 5,160 |
| min_mark_change_2q | 4,985 |
| mark_dispersion | 4,337 |
| min_mark_drift | 4,328 |
| mark_drift_2q | 4,034 |
| interest_rate | 3,934 |
| mark_drift | 3,725 |

The consensus mark across lenders (`avg_mark_across_lenders`) is the strongest predictor -- what the group thinks matters more than what any single lender reports.

---

## File Map

```
run_pipeline.py              Single-command pipeline orchestrator
filter_bdc.py                Shrinks SEC zips into data/raw/soi_all_bdcs.parquet

src/
  ingest_sec_bdc.py          Real SEC data ingestion and cleaning
  entity_resolution.py       Borrower name matching (normalize + hash)
  signals.py                 Mark drift, dispersion, cross-lender signals
  model.py                   LightGBM model (time-split, deterministic)
  newly_distressed.py        Rolling backtest + laggard alert
  app.py                     Streamlit dashboard (7 tabs)

data/
  raw/                       soi_all_bdcs.parquet (SEC filings)
  processed/                 warehouse.duckdb (all tables)
  sample/                    Synthetic sample for testing

models/
  lightgbm_model.txt         Trained model
  metrics.json               Model evaluation metrics
  newly_distressed_metrics.json   Rolling backtest + alert results
  laggard_alerts_latest.csv  Current quarter alerts

tests/                       pytest suite
docs/                        This file, loom script
```

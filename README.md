# Private Credit Early Warning Radar

An early warning system for private credit, built on real public SEC filings. It reads every loan that US Business Development Companies (BDCs) report each quarter, matches the same borrower across lenders, and flags loans that are likely to be marked down next quarter.

**Disclaimer:** Independent research project using public SEC filings. Not affiliated with any company mentioned. Not investment advice.

## The idea in one picture

Imagine 16 friends each lent money to the same person, and every quarter each friend writes down what they think their loan is worth. If one friend suddenly writes "worth 89" while the others still write "worth 97", the others often follow. This project watches for that moment.

## Real case: Pluralsight

Pluralsight's loan was held by 16 BDCs. From the real filings in this project:

| Quarter | Ares | Blue Owl | Golub | What happened |
|---|---|---|---|---|
| Q3 2023 | 0.94 | 0.97 | 1.00 | Ares starts cutting |
| Q4 2023 | 0.89 | 0.97 | 1.00 | Ares cuts again, most others don't move |
| Q1 2024 | 0.85 | 0.84 | 0.98 | Laggards start to follow |
| Q2 2024 | 0.48 | 0.47 | 0.54 | Everyone marks it down by half |

Marks are fair value divided by cost.

The **laggard alert** in this project fired in Q4 2023 for 10 lenders that had not yet moved. **7 of those 10 cut their marks by 5+ points the next quarter**, two quarters before the collapse.

## Data

Source: [SEC DERA BDC Data Sets](https://www.sec.gov/data-research/sec-markets-data/bdc-data-sets), `soi.tsv` (one row per holding).

| | |
|---|---|
| Files used | 7 quarterly zips, 2023 Q2 to 2024 Q4 |
| Raw rows | 730,310 |
| Clean debt positions | 163,539 |
| Lenders | 139 BDCs, grouped into managers |
| Borrowers | 16,717 raw names resolved to 13,251 |
| Period covered | Q4 2022 to Q3 2024 |

Cleaning problems found in the real data and fixed in `src/ingest_sec_bdc.py`:

1. Fair value and cost sit under mislabelled columns in the SEC file.
2. Each filing repeats prior periods, so only each filing's own period is kept.
3. Amended filings replace originals.
4. Lenders are keyed by CIK, so Owl Rock and Blue Owl are the same lender.
5. Subtotal and cash rows are removed by pattern and by size.
6. Borrower names are parsed from free text, and category prefixes like "Internet Software and Services" are learned from the data and stripped.
7. Zero cost rows such as undrawn revolvers are dropped.

## Results (rolling, out of sample)

Every quarter, models train only on earlier quarters. Score is PR-AUC, higher is better. Only loans that are healthy today (mark at or above 0.80) are scored.

**Target A: loan crosses below 0.80 next quarter**

| Method | PR-AUC |
|---|---|
| Rule: lowest mark first | **0.213** |
| LightGBM, original features | 0.175 |
| LightGBM, cross lender features | 0.187 |

The simple rule wins. This target is almost mechanically tied to how close the mark already is to 0.80.

**Target B: loan is marked down 5+ points next quarter** (base rate about 3%)

| Method | PR-AUC |
|---|---|
| Rule: lowest mark first | 0.098 |
| LightGBM, original features | 0.192 |
| LightGBM, cross lender features | **0.205** |

The model is about 6x better than random and 2x better than the rule. Cross lender features add +0.013 (90% CI +0.003 to +0.023, resampling whole borrowers).

**Laggard alert:** another manager cut the borrower by 4+ points this quarter and this lender's mark moved less than 2 points.

| Others cut at least | Alerts | Hit rate | Lift |
|---|---|---|---|
| 3 points | 615 | 11.5% | 3.6x |
| **4 points** | **445** | **13.7%** | **4.3x** |
| 5 points | 348 | 12.6% | 3.9x |

Threshold chosen with Pluralsight excluded. Base rate 3.2%.

## Honest limitations

1. Non accrual status lives in filing footnotes, not tagged columns, so it is not captured yet.
2. Only about 7 quarters of history. The alert threshold was chosen on the same period it is tested on, so a stricter test would pick it on 2023 and check it on 2024.
3. Borrower name parsing is heuristic. Some names still carry noise.
4. Quarters after a restructuring are excluded from targets because old and new loans are not comparable.
5. A manager level version of the cross lender features (v2) did not improve the model, even though the lag effect itself is strong. It is rare, about 2% of rows, so it works better as a simple alert than as a model feature.

## How to run

```bash
pip install -r requirements.txt

# Optional: rebuild the data file from the 7 SEC zips in your Downloads folder
python filter_bdc.py

# Full pipeline: ingest, entity resolution, signals, model, backtest, alerts
python run_pipeline.py

# Dashboard
streamlit run src/app.py

# Tests
pytest tests/
```

`python run_pipeline.py --sample` runs on the old synthetic sample instead.

## Dashboard tabs

Market Overview, Borrower Drilldown (search any borrower, defaults to Pluralsight), Lender Comparison, Early Warning Watchlist, Laggard Alerts, Backtest Results, Architecture.

## Project structure

```
filter_bdc.py              Shrinks the SEC zips into data/raw/soi_all_bdcs.parquet
run_pipeline.py            One command pipeline
src/ingest_sec_bdc.py      Real data ingestion and cleaning
src/entity_resolution.py   Matches borrower names across lenders
src/signals.py             Mark drift, dispersion, PIK and par migration, watchlist
src/model.py               Original LightGBM model
src/newly_distressed.py    Rolling backtest, cross lender features, laggard alert
src/app.py                 Streamlit dashboard
src/ingest.py              Legacy companyfacts ingestion, kept for reference only
data/raw/                  Real data (Parquet)
data/sample/               Old synthetic sample
models/                    Model, metrics, backtest results, latest alerts
tests/                     Unit tests
```

## Tech stack

Python, pandas, DuckDB, LightGBM, scikit-learn, Streamlit, Plotly.

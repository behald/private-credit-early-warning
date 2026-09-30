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

By Q1 2024, all **15 Pluralsight lenders were marked down 5+ points**. The rolling backtest model ranked them in the 97th percentile for risk the quarter before the collapse.

## Data

Source: [SEC DERA BDC Data Sets](https://www.sec.gov/data-research/sec-markets-data/bdc-data-sets), `soi.tsv` (one row per holding).

| | |
|---|---|
| Files used | 7 quarterly zips, 2023 Q2 to 2024 Q4 |
| Raw rows | 730,310 |
| Clean debt positions | 161,017 |
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
8. Rows with fair value zero or mark (fair value / cost) outside [0.05, 2.0] are dropped as data errors. Positions with fair value zero or mark below 0.05 go into a separate written-off table. This removed about 2,500 rows, including 721 with fair value exactly zero and 708 negative marks from negative cost entries.

## Results (rolling, out of sample)

Every quarter, models train only on earlier quarters. Score is PR-AUC, higher is better. Only loans that are healthy today (mark at or above 0.80) are scored.

**Target A: loan crosses below 0.80 next quarter**

| Method | PR-AUC |
|---|---|
| Rule: lowest mark first | **0.227** |
| LightGBM, original features | 0.197 |
| LightGBM, cross lender features | 0.209 |

The simple rule wins. This target is almost mechanically tied to how close the mark already is to 0.80.

**Target B: loan is marked down 5+ points next quarter** (base rate about 3%)

| Method | PR-AUC |
|---|---|
| Rule: lowest mark first | 0.105 |
| LightGBM, original features | 0.146 |
| LightGBM, cross lender features | **0.158** |

The model is about 5x better than random and 1.5x better than the rule. Cross lender features add +0.012 (90% CI +0.004 to +0.021, resampling whole borrowers).

**Laggard alert:** another manager cut the borrower by 7+ points this quarter and this lender's mark moved less than 2 points. Threshold chosen on pre-2024 data, validated on 2024+.

| | Pre-2024 (training) | 2024+ (out of sample) |
|---|---|---|
| Alerts at 7 pts | 102 | 125 |
| Hit rate | 15.7% | 8.8% |
| Lift vs base rate | 5.7x | 3.0x |

Pluralsight excluded from threshold selection. Base rate ~3%.

## Honest limitations

1. Non accrual status lives in filing footnotes, not tagged columns, so it is not captured yet.
2. Only about 7 quarters of history. The alert threshold is chosen on pre-2024 data and validated on 2024+, but both periods are short.
3. Borrower name parsing is heuristic. Some names still carry noise.
4. Quarters after a restructuring are excluded from targets because old and new loans are not comparable.
5. A manager level version of the cross lender features (v2) did not improve the model, even though the lag effect itself is strong. It is rare, about 2% of rows, so it works better as a simple alert than as a model feature.
6. Rows with mark outside [0.05, 2.0] are excluded at staging (negative marks from negative cost, extreme marks from unit errors). Positions with fair value zero or mark below 0.05 go into a separate written-off table. The staging filter removes about 2,500 of 187,000 debt positions.

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

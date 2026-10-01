# LEARNING.md - Private Credit Early Warning Radar

## Stage 1: Ingestion

### What we built
Think of it like a librarian who visits the SEC's public filing cabinet every quarter, pulls out the "portfolio report" page from each BDC's filing, and neatly copies the loan details into a spreadsheet.

We built a rate-limited SEC EDGAR client that fetches XBRL company facts for 30+ BDCs. The XBRL data contains tagged financial facts including the Schedule of Investments (SOI) with fair value, cost, and investment attributes per position. We extract these into structured JSON and create a small sample fixture for testing.

### Why this design, and alternatives rejected
- **XBRL company facts API over full filing HTML parsing**: The company facts endpoint aggregates all tagged facts across filings. Parsing HTML tables would require per-BDC heuristics since formats differ.
- **JSON raw storage over direct DB load**: Raw JSON lets us re-process without re-downloading. SEC rate limits (10 req/s) make re-downloads expensive.

### Key files to read, in order
1. `src/ingest.py` - EDGAR client and SOI extraction
2. `data/sample/soi_sample.json` - what raw data looks like
3. `tests/test_ingest.py` - expected behavior

### 5 likely interview questions

**Q: How does SEC EDGAR structure BDC filing data?**
A: BDCs file 10-Qs and 10-Ks with a Schedule of Investments. Since ~2020 these are XBRL-tagged. The company facts API aggregates all facts by concept (like InvestmentOwnedAtFairValue) with dimension members identifying each investment.

**Q: What is InvestmentIdentifierAxis?**
A: An XBRL dimension axis identifying individual investments in the SOI. Each investment is a member, and facts like fair value and cost are reported per member per period.

**Q: How do you handle SEC rate limits?**
A: Max 10 req/s with a descriptive User-Agent. We throttle to 100ms minimum between requests and cache responses on disk.

**Q: Why store raw data before transforming?**
A: Immutable audit trail. If parsing logic has a bug, we fix the parser and re-run on cached data without hitting the API again.

**Q: What could go wrong?**
A: XBRL tagging quality varies. Some BDCs use custom extensions instead of standard US-GAAP concepts, which our parser would miss.

### What could break in production
- Concept changes: monitor record count per BDC per quarter, flag sudden drops.
- Rate limit violations: log HTTP 403/429 responses.
- Stale data: compare filing dates against expected quarterly schedule.

---

## Stage 2: dbt Staging Models

### What we built
Taking raw notes and organizing them into clean, standardized tables with consistent formats and completeness checks.

dbt project with DuckDB as the warehouse. Staging layer cleans raw SOI data: casts types, derives quarters from period end dates, filters records with missing critical fields. Schema tests enforce not-null and unique constraints.

### Why this design
- **dbt-duckdb over Postgres/Snowflake**: DuckDB runs in-process, zero infrastructure. On a 16GB laptop this is the right call. Migration to Snowflake later is a one-line config change.
- **Staging as views, marts as tables**: Standard dbt pattern. Views cost no storage and reflect latest raw data. Marts materialized for dashboard query speed.

### Key files
1. `transforms/dbt_project.yml`
2. `transforms/models/staging/stg_soi_positions.sql`
3. `transforms/models/staging/schema.yml`

### 5 likely interview questions

**Q: Why use dbt?**
A: Version-controlled SQL transforms, built-in testing (not_null, unique), documentation, dependency graph. Industry standard for analytics engineering.

**Q: What is the medallion architecture?**
A: Bronze (staging), silver (intermediate), gold (marts). Each layer adds quality and business logic.

**Q: How do you handle data quality in dbt?**
A: Schema tests on every model plus custom tests for business rules (fair value positive, quarter in range).

**Q: Why DuckDB?**
A: In-process OLAP database. Handles analytical queries on millions of rows without server setup. Reads Parquet natively.

**Q: What would you change for production scale?**
A: Swap DuckDB for Snowflake or BigQuery (one-line config). Add incremental materialization. Add freshness checks and alerting.

### What could break
- Schema drift in raw EDGAR data. dbt tests catch it.
- Duplicate records. Unique test on accession_number prevents it.

---

## Stage 3: Entity Resolution

### What we built
Think of it like a detective matching aliases. "Acme Widgets Co., LLC" at Ares Capital and "Acme Widgets Corporation" at Blackstone are the same borrower. We need to figure that out automatically.

We normalize names (strip legal suffixes like LLC/Corp, lowercase, remove punctuation), then block by first character and industry to reduce comparisons. Within blocks, we compute cosine similarity of sentence-transformer embeddings. Pairs above 0.85 similarity are clustered using union-find.

### Why this design
- **Blocking + embeddings over all-pairs comparison**: With 5,000+ unique names, all-pairs is O(n^2) = 25 million comparisons. Blocking cuts this to ~50,000 within-block pairs.
- **Embeddings over edit distance**: Edit distance fails on "Acme Widgets" vs "Widget Corp Acme" (word reordering). Embeddings capture semantic similarity.
- **Manual override table**: No ML system is perfect. A CSV override table lets you correct specific known matches.

### Key files
1. `src/entity_resolution.py`
2. `tests/test_entity_resolution.py`
3. `data/sample/soi_sample.json` (shows name variants for same borrower)

### 5 likely interview questions

**Q: Why not just fuzzy string matching (Levenshtein)?**
A: Fuzzy matching fails on word reordering ("Sunrise Medical" vs "Medical Sunrise") and abbreviations ("Intl" vs "International"). Embeddings handle both.

**Q: What is blocking in entity resolution?**
A: Grouping records by cheap attributes (first letter, industry) so you only compare within groups. Reduces O(n^2) to something manageable.

**Q: How do you measure entity resolution quality?**
A: Precision and recall on a hand-labeled set of 200 name pairs. Precision is the priority (bad merges corrupt all downstream analysis), target 95%+.

**Q: What is union-find?**
A: A data structure for tracking which elements belong to the same group. When we find A matches B and B matches C, union-find efficiently tells us A, B, C are all the same entity. With path compression, each lookup is nearly O(1).

**Q: What happens when entity resolution is wrong?**
A: A false positive (merging two different borrowers) corrupts the cross-lender analysis. A false negative (missing a match) means we undercount lender overlap. False positives are worse, so we tune for high precision.

### What could break
- New borrower name patterns the model has not seen.
- Industry metadata missing, weakening blocking.
- Detect by monitoring cluster sizes: a cluster with 50+ names is likely wrong.

---

## Stage 4: Panel and Signal Marts

### What we built
The quarterly panel is the core analytical table: every borrower x lender x quarter combination with mark (fair value / cost), PIK flag, non-accrual flag. On top of this we compute signals: mark drift (how much the mark changed), cross-lender dispersion (do lenders agree), PIK migration (did it switch to payment-in-kind), and lender lag (who marks down last).

### Why this design
- **SQL in dbt over Python**: Signal computation is aggregation and window functions. SQL is clearer, testable, and runs inside DuckDB without data transfer overhead.
- **Documented formulas in signal_formulas.yml**: Every signal has a formula, range, and interpretation. This is both for the README and for interview discussions.

### Key files
1. `transforms/models/marts/mart_signals.sql`
2. `transforms/models/marts/mart_watchlist.sql`
3. `transforms/models/marts/mart_lender_comparison.sql`
4. `transforms/signal_formulas.yml`

### 5 likely interview questions

**Q: What is mark drift and why does it matter?**
A: Mark drift = mark(t) - mark(t-1). Sustained negative drift means the lender is marking down. It predicts future non-accrual better than the mark level alone.

**Q: What does cross-lender mark dispersion reveal?**
A: When multiple lenders hold the same borrower but disagree on fair value by more than 10%, it means one lender is either slow to recognize losses or using different valuation methods. This is a signal of information asymmetry.

**Q: What is PIK (payment in kind)?**
A: Instead of paying cash interest, the borrower adds the interest to the principal. It is a restructuring tool that signals the borrower cannot service cash payments. Migration from cash-pay to PIK is a strong warning.

**Q: How do you avoid look-ahead bias in signals?**
A: Every signal uses only data from the current and prior quarters (LAG window function). The target variable uses LEAD to look one quarter ahead. Train/test split is strictly by time.

**Q: Why track lender lag?**
A: If a lender consistently marks down after peers, it may be slow to recognize losses. This has portfolio risk implications and is interesting to fund administrators.

### What could break
- Borrowers with only one lender have no cross-lender dispersion (null). Handle with COALESCE.
- Quarters with no data for a borrower-lender pair create gaps in the window functions.

---

## Stage 5: Early Warning Model

### What we built
A LightGBM classifier that predicts which positions become non-accrual or fall below 0.80 mark next quarter. The baseline is logistic regression on mark level alone. We compare using PR-AUC (because distress is rare, ~5% of positions), precision@50, and calibration.

### Why this design
- **LightGBM over deep learning**: Tabular data with <50 features. Gradient boosting consistently wins here. LightGBM is fast and handles missing values natively.
- **PR-AUC over accuracy or ROC-AUC**: With ~5% positive rate, a model predicting "no distress" always gets 95% accuracy. PR-AUC focuses on the rare positive class.
- **Time-based splits over random splits**: Financial data has time dependence. Random splits leak future information into training. We train on quarters 1-6, validate on 7, test on 8.

### Key files
1. `src/model.py`
2. `models/metrics.json` (after training)
3. `tests/test_model.py`

### 5 likely interview questions

**Q: Why PR-AUC instead of ROC-AUC?**
A: With severe class imbalance (~5% positive), ROC-AUC can look good even if precision is low. PR-AUC directly measures the trade-off between precision and recall on the minority class.

**Q: What is precision@50?**
A: If an analyst reviews the top 50 riskiest positions, what fraction are actually distressed? This is the most actionable metric: it measures whether the watchlist is useful.

**Q: How do you prevent data leakage?**
A: Three layers. (1) Features use only current and prior quarter data (window functions with LAG). (2) Train/test split is by time, never random. (3) Entity resolution happens before feature computation, not after.

**Q: What is the baseline and why do you need it?**
A: Logistic regression on mark alone. It answers "how much does the fancy model add beyond just looking at the current mark?" If LightGBM's PR-AUC is only marginally better, the complexity is not justified.

**Q: How would you deploy this model?**
A: Quarterly batch scoring. After each filing season, re-run the pipeline, retrain on the latest data, and update the watchlist. No need for real-time since filings are quarterly.

### What could break
- Concept drift: market regimes change. Monitor PR-AUC on each new quarter and retrain if it drops.
- Feature importance shift: if mark_dispersion stops being predictive, investigate whether BDC reporting practices changed.

---

## Stage 6: Dashboard

### What we built
Streamlit app with four tabs: Market Overview (trends), Borrower Drilldown (select a borrower, see marks across lenders), Lender Comparison (portfolio-level metrics), and Early Warning Watchlist (top 50 riskiest positions).

### Key files
1. `src/app.py`

### 5 likely interview questions

**Q: Why Streamlit over a custom React app?**
A: Speed to demo. Streamlit is Python-native, reads DuckDB directly, and deploys for free on Community Cloud. For a portfolio project, time-to-demo matters more than frontend polish.

**Q: How does company customization work?**
A: The `?company=slug` URL parameter loads a YAML profile that highlights specific views and filters data. For a BDC operator like BlackRock, it shows their BDC's peer comparison.

**Q: How do you handle the research disclaimer?**
A: Every page shows a disclaimer. The company brief explicitly states this is an independent project using public data. No claims about internal systems.

---

## Stage 7: Orchestration

### What we built
Dagster assets for each pipeline step. A quarterly schedule triggers at 6 AM on the 1st of Jan/Apr/Jul/Oct (after filing season). Falls back to plain Python script if Dagster is not installed.

### Key files
1. `src/orchestration.py`

### 5 likely interview questions

**Q: Why Dagster over Airflow?**
A: Dagster uses ~200MB RAM vs Airflow's ~1GB+ (scheduler + webserver + DB). On a 16GB laptop running Docker, this matters. Dagster's asset-based model also fits this pipeline better than Airflow's task-based DAGs.

**Q: What is an asset vs a task?**
A: A task is "run this function." An asset is "produce this dataset." Dagster's asset model lets you say "I want the watchlist to be fresh" and it figures out which upstream steps need to run.

---

## Stage 8: Deployment

### Key files
1. `DEPLOY.md` - exact steps
2. `docs/loom_script.md` - walkthrough script

Cost: $0/month (Streamlit Community Cloud + SEC free API).

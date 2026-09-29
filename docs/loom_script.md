# Loom Walkthrough Script: Private Credit Early Warning Radar

**Target length:** 60-90 seconds

## Script

**[0:00 - 0:10] Hook**
"Private credit is a $1.7 trillion market, but it's opaque. I built a system that uses public SEC filings to detect deteriorating borrowers before they default."

**[0:10 - 0:25] What it does**
"I pull Schedule of Investments data from 30 BDCs via SEC EDGAR, resolve the same borrower across different lenders despite name variations, and build a quarterly panel of marks, PIK flags, and non-accrual status."

**[0:25 - 0:40] Dashboard - Market Overview**
[Show the Market Overview tab]
"Here's the market overview. Average marks across the portfolio over time, with stress indicators by quarter."

**[0:40 - 0:55] Key insight**
[Switch to Borrower Drilldown, pick a borrower held by multiple lenders]
"This is where it gets interesting. This borrower is held by three BDCs. Two marked it down to 0.80, but this lender still holds it at 0.95. That dispersion is a signal."

**[0:55 - 1:10] Model**
[Switch to Watchlist]
"The early warning model uses mark drift, PIK migration, and cross-lender dispersion to predict which positions go non-accrual next quarter. PR-AUC of [X] with strict time-based splits."

**[1:10 - 1:20] Close**
"This is built with dbt, DuckDB, LightGBM, and Streamlit. The entity resolution achieves 95%+ precision on hand-labeled pairs. Full code on GitHub."

## Tips for recording
- Have the dashboard pre-loaded with data
- Use the borrower drilldown to show a real cross-lender discrepancy
- Mention the time-based validation explicitly -- interviewers care about this
- Keep it under 90 seconds. Busy managers will skip long videos.

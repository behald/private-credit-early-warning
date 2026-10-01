# Deployment Guide: Private Credit Early Warning Radar

## Live Demo

Streamlit Community Cloud reading a precomputed DuckDB file.

## Prerequisites

- GitHub account with the repo pushed
- Streamlit Community Cloud account (sign in with GitHub)
- Precomputed warehouse.duckdb under 500 MB

## Steps

### 1. Precompute the DuckDB file

```bash
cd 01-private-credit-radar

# Set up environment
cp ../.env.example .env
# Fill in SEC_USER_AGENT in .env

# Run the full pipeline
make setup
make data          # Downloads from SEC EDGAR (takes ~30 min)
make run           # Runs dbt models
python src/orchestration.py  # Or: make run

# The precomputed file is at data/processed/warehouse.duckdb
ls -lh data/processed/warehouse.duckdb
```

### 2. Prepare for Streamlit Cloud

```bash
# Create requirements.txt for Streamlit Cloud
cat > requirements.txt << 'EOF'
streamlit>=1.30
duckdb>=0.10
plotly>=5.0
pandas>=2.0
pyarrow>=14.0
pyyaml>=6.0
EOF

# Ensure the DuckDB file is small enough
# If over 500 MB, filter to recent quarters:
# python -c "import duckdb; con = duckdb.connect('data/processed/warehouse.duckdb'); con.execute('DELETE FROM raw_soi_positions WHERE fiscal_year < 2022')"
```

### 3. Push to GitHub

```bash
# Make sure .gitignore does NOT exclude the precomputed DuckDB for deployment
# Add the precomputed DuckDB to a separate branch or use Git LFS
git checkout -b deploy
git add data/processed/warehouse.duckdb
git add src/app.py requirements.txt
git commit -m "deploy: precomputed data for Streamlit Cloud"
git push origin deploy
```

### 4. Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io
2. Click "New app"
3. Select your GitHub repo, branch `deploy`, main file `src/app.py`
4. Click "Deploy"
5. The app will be live at `https://<your-app>.streamlit.app`

### 5. Test company customization

Visit: `https://<your-app>.streamlit.app/?company=blackrock`

This loads the company profile from `company_profiles/blackrock.yaml` and highlights relevant views.

## Alternative: Hugging Face Spaces

If Streamlit Cloud has issues:

1. Create a new Space on huggingface.co (Streamlit SDK)
2. Upload `src/app.py`, `data/processed/warehouse.duckdb`, `requirements.txt`
3. The Space will auto-deploy

## Monitoring

- Streamlit Cloud shows usage analytics
- The dashboard shows a "Data freshness" indicator based on the latest quarter in the data
- To refresh: re-run the pipeline, update the DuckDB file, push to GitHub

## Cost

- Streamlit Community Cloud: free
- SEC EDGAR API: free (respect rate limits)
- Total running cost: $0/month

# 🏦 Banking Fraud Detection & Customer Insights

An end-to-end project covering **data pipeline engineering, analytics, machine learning, and GenAI explainability** for digital banking.

## What it does
1. **ETL pipeline** (`src/pipeline.py`): loads raw transactions, removes duplicates, fixes missing values and inconsistent categories, engineers features (amount ratio, velocity, night flag, customer segment), and loads to **SQLite**.
2. **Analytics dashboard** (Streamlit): volume trends, spend by category, fraud by hour, customer segments.
3. **ML models** (`src/train.py`): Logistic Regression, Random Forest, Gradient Boosting compared on precision, recall, F1, ROC-AUC, PR-AUC (class imbalance handled with class weights).
4. **Fraud checker**: enter a transaction, get a risk score plus plain-English reason codes.
5. **Optional GenAI layer** (`src/explain.py`): if `ANTHROPIC_API_KEY` is set, Claude rewrites the reason codes into an analyst-friendly explanation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
First launch automatically generates data and trains models (about a minute). To run steps manually:
```bash
python -m src.pipeline
python -m src.train
```

## Deploy free on Streamlit Community Cloud
1. Push this folder to a **GitHub repo**.
2. Go to https://share.streamlit.io, click **New app**, pick the repo, set main file to `app.py`.
3. (Optional) Under **Settings > Secrets**, add: `ANTHROPIC_API_KEY = "your-key"` to enable the GenAI explanations.
4. Deploy. The app builds its own data and model on first load.

## Project structure
```
app.py            Streamlit app (dashboard, fraud checker, metrics)
src/data_gen.py   Synthetic dirty banking data
src/pipeline.py   Extract, transform, load to SQLite
src/train.py      Model training and comparison
src/explain.py    Reason codes + optional LLM explanation
```

## Notes
- Data is **synthetic**; no real customer data is used (important in banking for privacy).
- Accuracy is deliberately not reported: with ~2% fraud it is misleading.
- Swap in the Kaggle "Credit Card Fraud" dataset by replacing `data/raw_transactions.csv` with the same column names.

## Ideas to extend
SHAP explanations, threshold tuning by business cost, Docker, scheduled pipeline with Airflow/GitHub Actions, PostgreSQL instead of SQLite.

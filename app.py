import json
import sqlite3

import joblib
import pandas as pd
import streamlit as st

from src import pipeline, train
from src.data_gen import CATEGORIES, CHANNELS
from src.explain import llm_explanation, reason_codes

st.set_page_config(page_title="Bank Fraud & Insights", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="First run: building data and training models (about a minute)...")
def bootstrap():
    if not pipeline.DB.exists():
        pipeline.run()
    if not (train.MODEL_DIR / "model.joblib").exists():
        train.run()
    model = joblib.load(train.MODEL_DIR / "model.joblib")
    metrics = json.loads((train.MODEL_DIR / "metrics.json").read_text())
    return model, metrics


@st.cache_data
def load_data() -> pd.DataFrame:
    with sqlite3.connect(pipeline.DB) as con:
        return pd.read_sql("SELECT * FROM transactions", con, parse_dates=["timestamp"])


model, metrics = bootstrap()
df = load_data()

st.title("🏦 Banking Fraud Detection & Customer Insights")
tab1, tab2, tab3 = st.tabs(["📊 Insights", "🕵️ Fraud Checker", "🤖 Model Performance"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Customers", f"{df.customer_id.nunique():,}")
    c3.metric("Total volume", f"₹{df.amount.sum():,.0f}")
    c4.metric("Fraud rate", f"{df.is_fraud.mean():.2%}")

    a, b = st.columns(2)
    a.subheader("Monthly transaction volume")
    a.line_chart(df.groupby("month")["amount"].sum())
    b.subheader("Spend by category")
    b.bar_chart(df.groupby("category")["amount"].sum().sort_values(ascending=False))

    a, b = st.columns(2)
    a.subheader("Transactions by hour of day")
    a.bar_chart(df.groupby("hour").size())
    b.subheader("Fraud rate by hour of day")
    b.bar_chart(df.groupby("hour")["is_fraud"].mean())

    st.subheader("Customer segments")
    seg = df.groupby("segment").agg(customers=("customer_id", "nunique"), transactions=("txn_id", "count"),
                                    total_spend=("amount", "sum"), fraud_rate=("is_fraud", "mean")).round(3)
    st.dataframe(seg)

with tab2:
    st.write("Enter transaction details to get a fraud risk score and an explanation.")
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount (₹)", 1.0, 1_000_000.0, 12000.0, step=500.0)
    cust_avg = c1.number_input("Customer's typical spend (₹)", 1.0, 500_000.0, 1500.0, step=100.0)
    hour = c2.slider("Hour of day", 0, 23, 2)
    txn_day = c2.slider("Transactions by this customer today", 1, 15, 1)
    category = c3.selectbox("Category", CATEGORIES, index=4)
    channel = c3.selectbox("Channel", CHANNELS, index=1)
    foreign = c3.checkbox("Foreign transaction", value=True)

    if st.button("Analyse transaction", type="primary"):
        row = dict(amount=amount, amount_ratio=amount / cust_avg, hour=hour, is_night=int(0 <= hour <= 5),
                   is_foreign=int(foreign), txn_count_day=txn_day, customer_avg=cust_avg,
                   category=category, channel=channel)
        score = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
        level = "🔴 HIGH" if score >= 0.6 else "🟠 MEDIUM" if score >= 0.25 else "🟢 LOW"
        st.metric("Fraud risk score", f"{score:.1%}", level)
        reasons = reason_codes(row)
        st.subheader("Why?")
        for r in reasons:
            st.write(f"- {r}")
        text = llm_explanation(row, score, reasons)
        if text:
            st.info(text)

with tab3:
    st.write(f"Best model (by PR-AUC): **{metrics['best']}** | Fraud rate in data: {metrics['fraud_rate']:.2%}")
    st.dataframe(pd.DataFrame(metrics["results"]).T)
    st.caption("Accuracy is not shown on purpose: with ~2% fraud, a model that predicts 'never fraud' is 98% accurate. "
               "Precision, recall, and PR-AUC are what matter.")

"""ETL pipeline: raw CSV -> cleaned + feature-engineered table in SQLite."""
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.data_gen import generate

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_transactions.csv"
DB = ROOT / "data" / "bank.db"
log = logging.getLogger("pipeline")


def extract() -> pd.DataFrame:
    RAW.parent.mkdir(exist_ok=True)
    if not RAW.exists():
        log.info("No raw data found - generating synthetic data")
        generate().to_csv(RAW, index=False)
    df = pd.read_csv(RAW, parse_dates=["timestamp"])
    log.info("Extracted %d rows", len(df))
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset="txn_id").copy()
    df["category"] = df["category"].str.lower().str.strip()
    df["amount"] = df["amount"].fillna(df.groupby("customer_id")["amount"].transform("median"))
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0]
    log.info("Cleaning removed %d rows", n0 - len(df))

    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["customer_avg"] = df.groupby("customer_id")["amount"].transform("median")
    df["amount_ratio"] = df["amount"] / df["customer_avg"]
    df["txn_count_day"] = df.groupby(["customer_id", df["timestamp"].dt.date])["txn_id"].transform("count")

    # customer value segment by total spend
    spend = df.groupby("customer_id")["amount"].sum()
    seg = pd.qcut(spend, 3, labels=["low", "medium", "high"])
    df["segment"] = df["customer_id"].map(seg).astype(str)
    return df


def load(df: pd.DataFrame) -> None:
    with sqlite3.connect(DB) as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)
    log.info("Loaded %d rows into %s", len(df), DB)


def run() -> pd.DataFrame:
    df = transform(extract())
    load(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()

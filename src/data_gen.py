"""Generate a synthetic (dirty) banking transactions dataset."""
import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "fuel", "dining", "travel", "electronics", "utilities", "shopping", "atm"]
CHANNELS = ["pos", "online", "atm", "mobile"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n_customers=2000, n_txn=60000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cust_avg = rng.lognormal(mean=6.0, sigma=0.5, size=n_customers)  # typical spend per customer

    cid = rng.integers(1, n_customers + 1, n_txn)
    ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n_txn), unit="m")
    hour = ts.hour.values
    night = ((hour >= 0) & (hour <= 5)).astype(int)
    category = rng.choice(CATEGORIES, n_txn, p=[.22, .12, .15, .06, .07, .13, .18, .07])
    channel = rng.choice(CHANNELS, n_txn, p=[.4, .3, .1, .2])
    foreign = (rng.random(n_txn) < 0.05).astype(int)

    p_fraud = sigmoid(-5.0 + 1.4 * night + 1.7 * foreign + 0.9 * (channel == "online")
                      + 0.8 * np.isin(category, ["electronics", "travel"]))
    is_fraud = (rng.random(n_txn) < p_fraud).astype(int)

    amount = cust_avg[cid - 1] * rng.lognormal(0, 0.6, n_txn) * 0.4
    amount = np.where(is_fraud == 1, amount * rng.lognormal(1.6, 0.5, n_txn), amount)

    df = pd.DataFrame({
        "txn_id": np.arange(1, n_txn + 1), "customer_id": cid, "timestamp": ts,
        "amount": amount.round(2), "category": category, "channel": channel,
        "is_foreign": foreign, "is_fraud": is_fraud,
    })

    # Inject realistic dirt so the pipeline has real work to do
    dup = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.sample(frac=0.005, random_state=1).index, "amount"] = np.nan
    mask = df.sample(frac=0.03, random_state=2).index
    df.loc[mask, "category"] = df.loc[mask, "category"].str.upper()
    return df

"""Train and compare fraud models; save the best one."""
import json
import logging
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline import DB, ROOT

MODEL_DIR = ROOT / "models"
NUM = ["amount", "amount_ratio", "hour", "is_night", "is_foreign", "txn_count_day", "customer_avg"]
CAT = ["category", "channel"]
log = logging.getLogger("train")


def build(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def run():
    with sqlite3.connect(DB) as con:
        df = pd.read_sql("SELECT * FROM transactions", con)
    X, y = df[NUM + CAT], df["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                                min_samples_leaf=3, n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    }
    results, fitted = {}, {}
    for name, clf in models.items():
        pipe = build(clf).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "precision": round(precision_score(yte, pred, zero_division=0), 3),
            "recall": round(recall_score(yte, pred), 3),
            "f1": round(f1_score(yte, pred), 3),
            "roc_auc": round(roc_auc_score(yte, proba), 3),
            "pr_auc": round(average_precision_score(yte, proba), 3),
        }
        fitted[name] = pipe
        log.info("%s -> %s", name, results[name])

    best = max(results, key=lambda k: results[k]["pr_auc"])  # PR-AUC suits imbalanced data
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"best": best, "results": results,
                                                        "fraud_rate": round(float(y.mean()), 4)}, indent=2))
    log.info("Best model: %s", best)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()

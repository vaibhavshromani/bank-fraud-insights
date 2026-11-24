"""Plain-English explanations: rule-based reason codes, optionally polished by Claude."""
import os

import requests


def reason_codes(row: dict) -> list[str]:
    reasons = []
    if row["amount_ratio"] >= 3:
        reasons.append(f"Amount is {row['amount_ratio']:.1f}x the customer's typical spend")
    if row["is_night"]:
        reasons.append(f"Transaction happened at an unusual hour ({row['hour']}:00)")
    if row["is_foreign"]:
        reasons.append("Transaction originated abroad")
    if row["channel"] == "online":
        reasons.append("Card-not-present (online) transaction")
    if row["category"] in ("electronics", "travel"):
        reasons.append(f"High-risk merchant category ({row['category']})")
    if row["txn_count_day"] >= 5:
        reasons.append(f"{row['txn_count_day']} transactions by this customer today (high velocity)")
    return reasons or ["No strong individual risk signals; score driven by combined patterns"]


def llm_explanation(row: dict, score: float, reasons: list[str]) -> str | None:
    """Optional: needs ANTHROPIC_API_KEY. Returns None if unavailable."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    prompt = (f"A bank fraud model gave this transaction a risk score of {score:.0%}.\n"
              f"Signals: {'; '.join(reasons)}\n"
              "Write a 2-sentence plain-English explanation for a bank analyst. Do not invent facts.")
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"), "max_tokens": 200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except Exception:
        return None

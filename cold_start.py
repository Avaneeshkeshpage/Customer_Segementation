"""
cold_start.py
-----------------
Classifies a customer into one of three states before deciding whether it's
safe to trust their K-Means segment assignment:

  - "no_history"  : customer_id has zero transactions (e.g. just signed up)
  - "cold_start"   : customer has some transactions, but too few / too recent
                      to make RFM values reliable yet
  - "established"  : enough history to trust the real K-Means segment

This directly implements the cold-start handling flowchart from the project
design: new users get a fixed onboarding action instead of being force-fit
into a behavioral cluster based on almost no data.
"""

import pandas as pd


def classify_customer_status(
    customer_id: str,
    transactions: pd.DataFrame,
    min_transactions: int = 2,
    recency_threshold_days: int = 14,
) -> str:
    """
    Parameters
    ----------
    customer_id : the customer to check
    transactions : the raw transactions DataFrame (customer_id, transaction_date, amount)
    min_transactions : fewer orders than this counts as "cold start", not "established"
    recency_threshold_days : only treat a low-order customer as "cold start" if their
                              last purchase was this recent (otherwise they're just a
                              genuinely low-frequency established customer, e.g. "At Risk")

    Returns
    -------
    "no_history" | "cold_start" | "established"
    """
    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    customer_txns = df[df["customer_id"] == customer_id]

    if len(customer_txns) == 0:
        return "no_history"

    reference_date = df["transaction_date"].max()
    days_since_last = (reference_date - customer_txns["transaction_date"].max()).days
    n_orders = len(customer_txns)

    if n_orders < min_transactions and days_since_last <= recency_threshold_days:
        return "cold_start"

    return "established"

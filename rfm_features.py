"""
rfm_features.py
-----------------
Converts raw transaction-level data into customer-level RFM features:
  - Recency:   days since the customer's last purchase
  - Frequency: total number of purchases
  - Monetary:  total amount spent

This is the core feature engineering step used by both train_model.py
and app.py, so it lives in one shared module.
"""

import pandas as pd
from datetime import datetime


def compute_rfm(transactions: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    transactions : DataFrame with columns [customer_id, transaction_date, amount]
    reference_date : the "today" date RFM is calculated against.
                      Defaults to one day after the latest transaction in the data.

    Returns
    -------
    DataFrame indexed by customer_id with columns [Recency, Frequency, Monetary]
    """
    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    if reference_date is None:
        reference_date = df["transaction_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        Recency=("transaction_date", lambda x: (reference_date - x.max()).days),
        Frequency=("transaction_date", "count"),
        Monetary=("amount", "sum"),
    ).reset_index()

    return rfm

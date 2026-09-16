"""
generate_data.py
-----------------
Generates a synthetic e-commerce transactions dataset so the project can be
run end-to-end without needing a real customer database.

Output: data/transactions.csv
Columns: customer_id, transaction_date, amount
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

TODAY = datetime(2026, 8, 15)
N_CUSTOMERS = 500
OUTPUT_DIR = "data"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "transactions.csv")


def generate_customer_group(n, recency_range, freq_range, amount_range, start_id):
    """Generate transactions for one behavioral group of customers."""
    rows = []
    for i in range(n):
        customer_id = f"CUST{start_id + i:04d}"
        n_orders = np.random.randint(*freq_range)
        for _ in range(n_orders):
            days_ago = np.random.randint(*recency_range)
            txn_date = TODAY - timedelta(days=int(days_ago))
            amount = round(np.random.uniform(*amount_range), 2)
            rows.append([customer_id, txn_date.strftime("%Y-%m-%d"), amount])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []

    # Champions: recent, frequent, high spend
    all_rows += generate_customer_group(
        n=120, recency_range=(0, 20), freq_range=(8, 20),
        amount_range=(800, 3000), start_id=1
    )

    # At-risk: not recent, used to buy often, moderate spend
    all_rows += generate_customer_group(
        n=130, recency_range=(90, 200), freq_range=(4, 12),
        amount_range=(300, 1200), start_id=121
    )

    # New customers: very recent, low frequency, low-to-moderate spend
    all_rows += generate_customer_group(
        n=150, recency_range=(0, 15), freq_range=(1, 3),
        amount_range=(100, 600), start_id=251
    )

    # Occasional / low-value: infrequent, low spend, moderate recency
    all_rows += generate_customer_group(
        n=100, recency_range=(30, 150), freq_range=(1, 4),
        amount_range=(50, 400), start_id=401
    )

    df = pd.DataFrame(all_rows, columns=["customer_id", "transaction_date", "amount"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} transactions for {df['customer_id'].nunique()} customers")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

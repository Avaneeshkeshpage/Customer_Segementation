# AI-Powered Customer Segmentation for Personalized Marketing

Unsupervised machine learning (K-Means) system that groups customers into
behavioral segments using RFM (Recency, Frequency, Monetary) features, then
recommends a personalized marketing action for each segment — deployed as an
interactive web app with Streamlit.

## Features

- **RFM-based K-Means segmentation** with automatic k selection via Elbow Method and Silhouette Score
- **DBSCAN validation baseline** — compares silhouette scores and flags outlier customers K-Means may have force-fit into a cluster
- **SHAP-based explainability** — for any individual customer, shows which RFM feature most influenced their segment assignment (via a surrogate Random Forest model)
- **Cold-start handling** — customers with no history or very few recent orders are detected and routed to a fixed onboarding action instead of being force-fit into a real K-Means segment
- **Loyalty points & tiers** — every purchase earns points (₹100 = 1 point), with a segment-based multiplier (Champions 2x, Loyal Customers 1.5x, etc.), rolling up into Bronze/Silver/Gold/Platinum tiers and a live leaderboard
- **Segment filtering, search, and export** — browse customers by segment or ID and download filtered CSVs
- **Interactive Streamlit dashboard** with 2D cluster visualization, segment profiling, and personalized marketing action recommendations

```
segmentation_project/
├── generate_data.py       # Creates a synthetic transactions dataset
├── rfm_features.py        # Computes RFM features from raw transactions
├── train_model.py         # Trains K-Means, compares with DBSCAN, builds SHAP explainer
├── marketing_actions.py   # Maps each segment to a recommended action
├── explainability.py      # Per-customer SHAP explanation helper
├── cold_start.py          # Detects new/low-history customers before trusting their segment
├── loyalty_points.py      # Points-per-purchase and tier calculation logic
├── app.py                 # Streamlit web app (the deployable interface)
├── requirements.txt       # Python dependencies
├── data/
│   └── transactions.csv   # Generated sample dataset
└── model/                 # Saved model artifacts (created after training)
    ├── scaler.pkl
    ├── kmeans_model.pkl
    ├── cluster_names.pkl
    ├── surrogate_model.pkl
    ├── dbscan_comparison.pkl
    ├── segmented_customers.csv
    └── k_selection.png
```

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Generate the sample dataset** (skip this if you have your own transactions CSV)
```bash
python generate_data.py
```

**3. Train the model**
```bash
python train_model.py
```
This computes RFM features, tests k = 2 to 10 using the Elbow Method and
Silhouette Score, fits the final K-Means model, profiles each cluster, and
saves everything to the `model/` folder.

**4. Launch the app**
```bash
streamlit run app.py
```
This opens the dashboard at `http://localhost:8501`.

## Using Your Own Data

Your CSV needs exactly these columns:

| Column | Type | Example |
|---|---|---|
| `customer_id` | text | CUST0001 |
| `transaction_date` | date (YYYY-MM-DD) | 2026-03-15 |
| `amount` | number | 799.00 |

You can either replace `data/transactions.csv` and re-run `train_model.py`,
or upload your CSV directly in the app's sidebar (it will use the already-
trained model to assign segments to your data).

### Recalibrating loyalty tiers

The Bronze/Silver/Gold/Platinum thresholds in `loyalty_points.py` were
calibrated against the included sample dataset's points distribution. If you
swap in your own data, points values will likely be very different — re-run
the check below and adjust the `TIERS` list in `loyalty_points.py` so the
tiers actually spread customers meaningfully (rather than everyone landing
in Bronze or everyone landing in Platinum):

```python
import pandas as pd
from loyalty_points import total_points_for_customer

rfm = pd.read_csv("model/segmented_customers.csv")
rfm["Points"] = rfm.apply(lambda r: total_points_for_customer(r["Monetary"], r["Segment"]), axis=1)
print(rfm["Points"].quantile([0.5, 0.7, 0.9, 0.97]))
```

---

## Deploying It as a Live Website

The easiest free option for a Streamlit app is **Streamlit Community Cloud**.
Here are the exact steps:

### Step 1: Push the project to GitHub

```bash
cd segmentation_project
git init
git add .
git commit -m "Initial commit: customer segmentation project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

> Make sure `data/transactions.csv` and the `model/` folder (with the .pkl
> files) are committed too — the deployed app needs them to run. Don't add
> a `.gitignore` rule that excludes them.

### Step 2: Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **"Deploy"**.

Streamlit Cloud will automatically:
- Read `requirements.txt` and install all dependencies
- Run `streamlit run app.py`
- Give you a live public URL like `https://your-app-name.streamlit.app`

### Step 3: Verify the live deployment

- Open the generated URL and confirm the dashboard loads with the sample data.
- Test the CSV upload feature with a small sample file.
- Share the link — anyone can now access your project without installing anything.

### Redeploying after changes

Any time you `git push` new changes to the `main` branch, Streamlit Community
Cloud automatically redeploys the app within a minute or two — no manual
redeploy step needed.

### Alternative deployment options

| Platform | Notes |
|---|---|
| **Streamlit Community Cloud** | Free, easiest, recommended for this project |
| **Hugging Face Spaces** | Free, also supports Streamlit apps directly |
| **Render / Railway** | Free tier available, slightly more setup (needs a `Procfile`) |

For an idea lab submission, Streamlit Community Cloud is the fastest path
from "working code" to "shareable live link."

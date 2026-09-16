"""
app.py
-----------------
Streamlit web app for the AI-Powered Customer Segmentation project.

Lets a user:
  - Use the built-in sample dataset, or upload their own transactions CSV
  - See RFM features computed automatically
  - View clusters visualized in 2D (PCA) and as a profile table
  - See the recommended marketing action for each segment
  - Look up an individual customer's assigned segment

Run locally: streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from rfm_features import compute_rfm
from marketing_actions import get_action, get_new_customer_action
from explainability import get_customer_explanation
from cold_start import classify_customer_status
from loyalty_points import total_points_for_customer, calculate_points, get_tier

MODEL_DIR = "model"
FEATURES = ["Recency", "Frequency", "Monetary"]

st.set_page_config(page_title="Customer Segmentation", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    cluster_names = joblib.load(os.path.join(MODEL_DIR, "cluster_names.pkl"))
    surrogate_model = joblib.load(os.path.join(MODEL_DIR, "surrogate_model.pkl"))
    dbscan_comparison = joblib.load(os.path.join(MODEL_DIR, "dbscan_comparison.pkl"))
    return scaler, kmeans, cluster_names, surrogate_model, dbscan_comparison


def assign_segments(transactions: pd.DataFrame, scaler, kmeans, cluster_names) -> pd.DataFrame:
    rfm = compute_rfm(transactions)
    X_scaled = scaler.transform(rfm[FEATURES])
    rfm["Cluster"] = kmeans.predict(X_scaled)
    rfm["Segment"] = rfm["Cluster"].map(cluster_names)
    rfm["Points"] = rfm.apply(lambda r: total_points_for_customer(r["Monetary"], r["Segment"]), axis=1)
    rfm["Tier"] = rfm["Points"].apply(get_tier)
    return rfm


def main():
    st.title("🎯 AI-Powered Customer Segmentation")
    st.caption("Unsupervised learning (K-Means) for personalized marketing")

    if not os.path.exists(os.path.join(MODEL_DIR, "kmeans_model.pkl")):
        st.error(
            "No trained model found. Run `python generate_data.py` and then "
            "`python train_model.py` first to create the model artifacts."
        )
        st.stop()

    scaler, kmeans, cluster_names, surrogate_model, dbscan_comparison = load_artifacts()

    # --- Sidebar: data source ---
    st.sidebar.header("Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload transactions CSV (customer_id, transaction_date, amount)", type="csv"
    )

    if uploaded_file is not None:
        transactions = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(transactions)} transactions")
    else:
        transactions = pd.read_csv("data/transactions.csv")
        st.sidebar.info("Using built-in sample dataset")

    rfm = assign_segments(transactions, scaler, kmeans, cluster_names)

    # --- Overview metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(rfm))
    col2.metric("Segments Found", rfm["Segment"].nunique())
    col3.metric("Avg. Spend per Customer", f"₹{rfm['Monetary'].mean():,.0f}")

    st.divider()

    # --- Model validation: elbow/silhouette plot + DBSCAN comparison ---
    st.subheader("Model Selection & Validation")
    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        if os.path.exists(os.path.join(MODEL_DIR, "k_selection.png")):
            st.image(
                os.path.join(MODEL_DIR, "k_selection.png"),
                caption="Elbow Method and Silhouette Score used to choose k",
            )
        else:
            st.info("Run train_model.py to generate the k-selection plot.")

    with col_b:
        st.markdown("**K-Means vs DBSCAN (validation baseline)**")
        st.metric("K-Means Silhouette Score", dbscan_comparison["kmeans_silhouette"])
        db_score = dbscan_comparison["dbscan_silhouette"]
        st.metric("DBSCAN Silhouette Score", db_score if db_score is not None else "N/A")
        st.metric(
            "DBSCAN Outliers Flagged",
            f"{dbscan_comparison['dbscan_outliers_flagged']} ({dbscan_comparison['dbscan_outlier_pct']}%)",
        )
        st.caption(
            "K-Means was chosen as the primary model since it achieves a higher "
            "silhouette score. DBSCAN is run as a validation baseline and to "
            "flag potential outlier customers K-Means may have force-fit into a cluster."
        )

    st.divider()

    # --- Cluster visualization (PCA to 2D) ---
    st.subheader("Customer Segments (2D projection)")
    X_scaled = scaler.transform(rfm[FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    rfm["pca_x"], rfm["pca_y"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    segments = rfm["Segment"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for seg, color in zip(segments, colors):
        subset = rfm[rfm["Segment"] == seg]
        ax.scatter(subset["pca_x"], subset["pca_y"], label=seg, alpha=0.7, color=color)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    st.pyplot(fig)

    st.divider()

    # --- Cluster profile table ---
    st.subheader("Segment Profiles")
    profile = rfm.groupby("Segment")[FEATURES].mean().round(1)
    profile["Customer Count"] = rfm["Segment"].value_counts()
    st.dataframe(profile, use_container_width=True)

    st.divider()

    # --- Filter, search, and export by segment ---
    st.subheader("Browse & Export Customers")
    col_f1, col_f2 = st.columns([1, 1.5])
    with col_f1:
        selected_segments = st.multiselect(
            "Filter by segment",
            options=sorted(rfm["Segment"].unique()),
            default=sorted(rfm["Segment"].unique()),
        )
    with col_f2:
        search_id = st.text_input("Search by customer ID (optional)", "")

    filtered = rfm[rfm["Segment"].isin(selected_segments)]
    if search_id:
        filtered = filtered[filtered["customer_id"].str.contains(search_id, case=False, na=False)]

    st.dataframe(
        filtered[["customer_id", "Recency", "Frequency", "Monetary", "Segment"]],
        use_container_width=True,
        height=250,
    )
    st.download_button(
        f"⬇️ Download filtered customers ({len(filtered)} rows) as CSV",
        filtered.to_csv(index=False),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )

    st.divider()

    # --- Marketing actions per segment ---
    st.subheader("Recommended Marketing Actions")
    for seg in rfm["Segment"].unique():
        info = get_action(seg)
        with st.expander(f"📌 {seg}"):
            st.write(f"**Profile:** {info['description']}")
            st.write(f"**Recommended action:** {info['action']}")

    st.divider()

    # --- New customer / cold-start check ---
    st.subheader("🆕 Check a New Customer (Cold-Start Handling)")
    st.caption(
        "K-Means needs enough purchase history to be reliable. This check keeps very "
        "new or unseen customers out of the real segments and gives them a fixed "
        "onboarding action instead — rather than force-fitting them into a cluster "
        "based on almost no data."
    )
    check_id = st.text_input("Enter any customer ID to check (existing or made-up)", "")

    if check_id:
        status = classify_customer_status(check_id, transactions)

        if status in ("no_history", "cold_start"):
            info = get_new_customer_action(status)
            label = "No purchase history yet" if status == "no_history" else "Too new for reliable segmentation"
            st.warning(f"**Status:** {label}")
            st.write(f"**Why:** {info['description']}")
            st.info(f"**Recommended action:** {info['action']}")
        elif status == "established" and check_id in rfm["customer_id"].values:
            seg_row = rfm[rfm["customer_id"] == check_id].iloc[0]
            st.success(f"**Status:** Established customer — safe to use real segment")
            st.write(f"**Segment:** {seg_row['Segment']}")
        else:
            st.error("Could not determine status for this customer ID.")

    st.divider()

    # --- Loyalty points leaderboard ---
    st.subheader("🏆 Loyalty Points Leaderboard")
    st.caption(
        "Points are earned per purchase (₹100 spent = 1 point), with a multiplier "
        "based on segment — Champions earn 2x, Loyal Customers 1.5x, and so on. "
        "This turns the 'reward Champions' marketing action into an actual mechanic."
    )
    top_customers = rfm.sort_values("Points", ascending=False).head(10)
    st.dataframe(
        top_customers[["customer_id", "Segment", "Points", "Tier"]].reset_index(drop=True),
        use_container_width=True,
    )

    tier_counts = rfm["Tier"].value_counts().reindex(["Platinum", "Gold", "Silver", "Bronze"]).fillna(0)
    st.bar_chart(tier_counts)

    st.divider()

    # --- Individual customer lookup ---
    st.subheader("Look Up a Customer")
    customer_id = st.selectbox("Select a customer ID", rfm["customer_id"].sort_values())
    row = rfm[rfm["customer_id"] == customer_id].iloc[0]
    st.write(f"**Segment:** {row['Segment']}")
    st.write(f"**Recency:** {row['Recency']} days | **Frequency:** {row['Frequency']} orders | **Monetary:** ₹{row['Monetary']:,.0f}")
    st.write(f"**Loyalty Points:** {row['Points']:.1f} pts — **Tier:** 🏅 {row['Tier']}")
    action = get_action(row["Segment"])
    st.info(f"Suggested action: {action['action']}")

    with st.expander("💳 Simulate a new purchase for this customer"):
        new_amount = st.number_input(
            "Purchase amount (₹)", min_value=0.0, value=500.0, step=50.0, key=f"amt_{customer_id}"
        )
        if st.button("Record purchase", key=f"btn_{customer_id}"):
            earned = calculate_points(new_amount, row["Segment"])
            new_total = row["Points"] + earned
            new_tier = get_tier(new_total)
            st.success(f"Earned **{earned} points** for this ₹{new_amount:,.0f} purchase.")
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("New Points Total", f"{new_total:.1f}", delta=f"+{earned}")
            col_p2.metric("Tier", new_tier, delta=("↑ Upgraded!" if new_tier != row["Tier"] else None))
            st.caption(
                "Note: this is a live simulation for demo purposes — it does not "
                "persist to the saved dataset. In a production system, this would "
                "write to a transactions table and trigger a re-run of the pipeline."
            )

    with st.expander("🔍 Why was this customer placed in this segment?"):
        try:
            explanation = get_customer_explanation(surrogate_model, row, FEATURES, int(row["Cluster"]))
            st.caption(
                "SHAP values from a surrogate model trained to reproduce the K-Means "
                "assignment — positive values pushed the customer toward this segment, "
                "negative values pushed away from it."
            )
            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
            colors = ["#1baf7a" if v > 0 else "#eb6834" for v in explanation["Contribution"]]
            ax2.barh(explanation["Feature"], explanation["Contribution"], color=colors)
            ax2.set_xlabel("SHAP contribution")
            ax2.invert_yaxis()
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Explanation unavailable for this customer: {e}")

    st.divider()
    st.download_button(
        "⬇️ Download segmented customer data (CSV)",
        rfm.to_csv(index=False),
        file_name="segmented_customers.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

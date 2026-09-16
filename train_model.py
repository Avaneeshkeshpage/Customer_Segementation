"""
train_model.py
-----------------
End-to-end training pipeline:
  1. Load transactions
  2. Compute RFM features
  3. Scale features
  4. Test k = 2..10 using Elbow Method + Silhouette Score
  5. Fit final K-Means model
  6. Compare against DBSCAN as a validation baseline
  7. Profile clusters and assign persona names
  8. Build a SHAP surrogate model for per-customer explainability
  9. Save all model artifacts

Run: python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import shap

from rfm_features import compute_rfm

DATA_PATH = "data/transactions.csv"
MODEL_DIR = "model"
FINAL_K = 4  # chosen after reviewing elbow/silhouette results; adjust as needed


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Try a range of k values, return metrics for each so we can pick the best k."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        results.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        })
    return pd.DataFrame(results)


def plot_k_selection(metrics_df, save_path="model/k_selection.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#2E74B5")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="#1baf7a")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved k-selection plot to {save_path}")


def name_clusters(profile_df: pd.DataFrame) -> dict:
    """
    Assign a human-readable persona name to each cluster based on its
    average RFM profile.

    Approach: compute one composite "customer value" score per cluster from
    standardized R, F, M values (recent + frequent + high-spend = high score),
    then rank clusters from best to worst and assign names from an ordered
    persona list. This always produces distinct, sensible labels regardless
    of how many clusters k is set to.
    """
    z = (profile_df[["Recency", "Frequency", "Monetary"]] - profile_df[["Recency", "Frequency", "Monetary"]].mean()) \
        / profile_df[["Recency", "Frequency", "Monetary"]].std(ddof=0).replace(0, 1)

    # Recency is "bad" when high, so subtract it; Frequency/Monetary are "good" when high.
    score = z["Frequency"] + z["Monetary"] - z["Recency"]
    ranked_clusters = score.sort_values(ascending=False).index.tolist()

    persona_pool = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "At Risk",
        "New Customers",
        "Low-Value / Lapsed",
    ]

    names = {}
    for i, cluster_id in enumerate(ranked_clusters):
        names[cluster_id] = persona_pool[i] if i < len(persona_pool) else f"Segment {i + 1}"

    return names


def compare_with_dbscan(X_scaled, kmeans_labels, eps=0.3, min_samples=8):
    """
    Fit DBSCAN on the same scaled data and compare against the K-Means result.
    This is used as a validation baseline: DBSCAN doesn't need a predefined k
    and can flag outliers (label -1), which helps sanity-check the K-Means clusters.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_scaled)

    n_clusters_found = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int(np.sum(db_labels == -1))

    # Silhouette score is undefined if DBSCAN finds fewer than 2 clusters
    # or only noise, so guard against that.
    non_noise_mask = db_labels != -1
    if n_clusters_found >= 2 and non_noise_mask.sum() > n_clusters_found:
        db_silhouette = silhouette_score(X_scaled[non_noise_mask], db_labels[non_noise_mask])
    else:
        db_silhouette = None

    comparison = {
        "kmeans_silhouette": round(float(silhouette_score(X_scaled, kmeans_labels)), 3),
        "dbscan_silhouette": round(float(db_silhouette), 3) if db_silhouette is not None else None,
        "dbscan_clusters_found": n_clusters_found,
        "dbscan_outliers_flagged": n_noise,
        "dbscan_outlier_pct": round(100 * n_noise / len(db_labels), 1),
    }
    return comparison, db_labels


def build_explainer(rfm: pd.DataFrame, features: list):
    """
    K-Means clusters aren't directly explainable with SHAP (it's not a
    predictive classifier). The standard workaround: train a fast surrogate
    classifier (Random Forest) to predict "which cluster" from the RFM
    features, then run SHAP on THAT model. Since the surrogate is trained
    to mimic the clustering assignment, its SHAP values approximate why
    each customer landed in their cluster.
    """
    surrogate = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    surrogate.fit(rfm[features], rfm["Cluster"])

    explainer = shap.TreeExplainer(surrogate)
    return surrogate, explainer


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load transactions
    transactions = pd.read_csv(DATA_PATH)

    # 2. Compute RFM
    rfm = compute_rfm(transactions)
    print("Sample RFM data:")
    print(rfm.head())

    # 3. Scale features
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    # 4. Determine optimal k
    metrics_df = find_optimal_k(X_scaled)
    print("\nK selection metrics:")
    print(metrics_df.to_string(index=False))
    plot_k_selection(metrics_df)

    # 5. Fit final K-Means model
    kmeans = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    final_silhouette = silhouette_score(X_scaled, rfm["Cluster"])
    print(f"\nFinal model: k={FINAL_K}, silhouette score={final_silhouette:.3f}")

    # 6b. Compare against DBSCAN as a validation baseline
    comparison, db_labels = compare_with_dbscan(X_scaled, rfm["Cluster"].values)
    rfm["DBSCAN_Label"] = db_labels
    print("\nK-Means vs DBSCAN comparison:")
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    # 7. Profile clusters and assign persona names
    profile = rfm.groupby("Cluster")[features].mean()
    cluster_names = name_clusters(profile)
    rfm["Segment"] = rfm["Cluster"].map(cluster_names)

    print("\nCluster profile (mean RFM values):")
    profile["Segment"] = profile.index.map(cluster_names)
    print(profile)

    # 8. Build a SHAP explainer (surrogate model) for per-customer explanations
    surrogate_model, explainer = build_explainer(rfm, features)
    print("\nBuilt SHAP surrogate model for per-customer explainability")

    # 9. Save artifacts
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(cluster_names, os.path.join(MODEL_DIR, "cluster_names.pkl"))
    joblib.dump(surrogate_model, os.path.join(MODEL_DIR, "surrogate_model.pkl"))
    joblib.dump(comparison, os.path.join(MODEL_DIR, "dbscan_comparison.pkl"))
    rfm.to_csv(os.path.join(MODEL_DIR, "segmented_customers.csv"), index=False)

    print(f"\nSaved model artifacts to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()

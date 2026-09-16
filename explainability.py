"""
explainability.py
-----------------
Helper functions for explaining WHY a customer was assigned to their
segment, using SHAP values from the surrogate Random Forest model
saved during training.
"""

import numpy as np
import pandas as pd
import shap


def get_customer_explanation(surrogate_model, customer_row: pd.Series, features: list, cluster_id: int):
    """
    Returns a DataFrame of SHAP contribution values for one customer,
    explaining how much each RFM feature pushed them toward their
    assigned cluster.
    """
    explainer = shap.TreeExplainer(surrogate_model)
    X_customer = customer_row[features].values.reshape(1, -1)

    shap_values = explainer.shap_values(X_customer)

    # shap_values shape depends on sklearn/shap version: either a list per class
    # or a single 3D array (n_samples, n_features, n_classes). Handle both.
    if isinstance(shap_values, list):
        contributions = shap_values[cluster_id][0]
    else:
        contributions = shap_values[0, :, cluster_id]

    result = pd.DataFrame({
        "Feature": features,
        "Value": customer_row[features].values,
        "Contribution": contributions,
    }).sort_values("Contribution", key=abs, ascending=False)

    return result

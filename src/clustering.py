"""
clustering.py
-------------
K-Means training, evaluation helpers, and heuristic segment labelling.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.preprocessing import CLUSTERING_FEATURES

RANDOM_STATE = 42


# ── Elbow & Silhouette evaluation ───────────────────────────────────

def compute_elbow(X_scaled: np.ndarray, k_range: range | None = None) -> dict:
    """Compute inertia (WCSS) for each K in *k_range*.

    Returns  {"k_values": [...], "inertias": [...]}
    """
    if k_range is None:
        k_range = range(2, 11)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return {"k_values": list(k_range), "inertias": inertias}


def compute_silhouette_scores(X_scaled: np.ndarray, k_range: range | None = None) -> dict:
    """Compute silhouette score for each K in *k_range*.

    Returns  {"k_values": [...], "scores": [...]}
    """
    if k_range is None:
        k_range = range(2, 11)
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))
    return {"k_values": list(k_range), "scores": scores}


# ── Model training ──────────────────────────────────────────────────

def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """Fit a KMeans model and return it."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    model.fit(X_scaled)
    return model


# ── Segment labelling (heuristic) ───────────────────────────────────

def label_clusters(df: pd.DataFrame, labels: np.ndarray, k: int) -> dict:
    """Assign a human-readable business label to each cluster.

    The labelling uses actual cluster statistics and a simple priority
    scoring scheme so that the label always reflects the data, not
    a hardcoded mapping.

    Returns
    -------
    dict  mapping  cluster_id (int) → {"label": str, "recommendation": str}
    """
    df = df.copy()
    df["Cluster"] = labels

    # Compute per-cluster means on original (unscaled) features
    agg = df.groupby("Cluster")[CLUSTERING_FEATURES].mean()

    # Normalise each statistic to [0, 1] across clusters for comparison
    normed = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)

    # Composite scores (higher ⇒ "better" customer from a revenue viewpoint)
    # spending_score: combines spending, purchases, order value
    # engagement_score: website visits and recency (low days since last purchase ⇒ higher engagement)
    # income_score: annual income

    spending_score   = normed["TotalSpending"] * 0.5 + normed["NumPurchases"] * 0.3 + normed["AvgOrderValue"] * 0.2
    engagement_score = normed["WebsiteVisitsPerMonth"] * 0.6 + (1 - normed["DaysSinceLastPurchase"]) * 0.4
    income_score     = normed["AnnualIncome"]

    composite = spending_score * 0.40 + engagement_score * 0.35 + income_score * 0.25

    # Sort clusters by composite score descending
    ranked = composite.sort_values(ascending=False).index.tolist()

    # Map ordered rank → label
    label_pool = [
        ("High-Value Customers",
         "Provide VIP programmes, exclusive deals, and personalised recommendations to retain these top customers."),
        ("Loyal Customers",
         "Reward loyalty with cashback, early-access sales, and referral bonuses to keep engagement high."),
        ("Potential Customers",
         "Target with upselling campaigns, personalised product suggestions, and limited-time offers."),
        ("Occasional Shoppers",
         "Re-engage with reminder emails, abandoned-cart nudges, and seasonal discount campaigns."),
        ("At-Risk Customers",
         "Launch win-back campaigns with special discounts, feedback surveys, and personalised outreach."),
    ]

    # If k > len(label_pool), extend with generic labels
    while len(label_pool) < k:
        idx = len(label_pool) + 1
        label_pool.append(
            (f"Segment {idx}",
             "Analyse this segment further to design targeted marketing strategies.")
        )

    result = {}
    for rank, cluster_id in enumerate(ranked):
        label, rec = label_pool[rank]
        result[int(cluster_id)] = {"label": label, "recommendation": rec}
    return result


def get_cluster_summary(df: pd.DataFrame, labels: np.ndarray, cluster_labels: dict) -> pd.DataFrame:
    """Return a summary DataFrame with one row per cluster."""
    df = df.copy()
    df["Cluster"] = labels

    agg = df.groupby("Cluster").agg(
        CustomerCount=("Cluster", "size"),
        AvgAnnualIncome=("AnnualIncome", "mean"),
        AvgTotalSpending=("TotalSpending", "mean"),
        AvgNumPurchases=("NumPurchases", "mean"),
        AvgOrderValue=("AvgOrderValue", "mean"),
        AvgDaysSinceLastPurchase=("DaysSinceLastPurchase", "mean"),
    ).reset_index()

    # Round for readability
    for col in agg.columns:
        if agg[col].dtype in ("float64", "float32"):
            agg[col] = agg[col].round(2)

    agg["SegmentLabel"] = agg["Cluster"].map(lambda c: cluster_labels[c]["label"])
    agg["Recommendation"] = agg["Cluster"].map(lambda c: cluster_labels[c]["recommendation"])

    return agg

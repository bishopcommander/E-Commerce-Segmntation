"""
visualization.py
----------------
Chart-building helpers that return Plotly figure objects for the Streamlit
dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Colour palette (accessible, vibrant)
PALETTE = [
    "#6366F1",  # indigo
    "#F59E0B",  # amber
    "#10B981",  # emerald
    "#EF4444",  # red
    "#3B82F6",  # blue
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F97316",  # orange
    "#06B6D4",  # cyan
]


def _colour_map(labels: np.ndarray, cluster_labels: dict) -> dict:
    unique = sorted(set(labels))
    return {cluster_labels[c]["label"]: PALETTE[i % len(PALETTE)] for i, c in enumerate(unique)}


# ── Elbow chart ──────────────────────────────────────────────────────

def plot_elbow(elbow_data: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=elbow_data["k_values"],
        y=elbow_data["inertias"],
        mode="lines+markers",
        marker=dict(size=10, color="#6366F1"),
        line=dict(width=3, color="#6366F1"),
        name="Inertia (WCSS)",
    ))
    fig.update_layout(
        title="Elbow Method – Inertia vs Number of Clusters (K)",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (WCSS)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── Silhouette chart ────────────────────────────────────────────────

def plot_silhouette(sil_data: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sil_data["k_values"],
        y=sil_data["scores"],
        marker_color="#10B981",
        name="Silhouette Score",
    ))
    fig.update_layout(
        title="Silhouette Score for Different K Values",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── PCA 2-D scatter ─────────────────────────────────────────────────

def plot_pca_clusters(X_scaled: np.ndarray, labels: np.ndarray, cluster_labels: dict) -> go.Figure:
    """Project scaled features to 2D with PCA and colour by cluster label."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    segment_names = [cluster_labels[l]["label"] for l in labels]

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Segment": segment_names,
    })

    colour_map = _colour_map(labels, cluster_labels)

    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Segment",
        color_discrete_map=colour_map,
        title="Customer Segments – PCA Projection",
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        legend_title_text="Segment",
    )
    return fig


# ── Cluster distribution (pie / donut) ──────────────────────────────

def plot_cluster_distribution(labels: np.ndarray, cluster_labels: dict) -> go.Figure:
    counts = pd.Series(labels).value_counts().sort_index()
    names = [cluster_labels[c]["label"] for c in counts.index]
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]

    fig = go.Figure(go.Pie(
        labels=names,
        values=counts.values,
        hole=0.45,
        marker=dict(colors=colours),
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig.update_layout(
        title="Customer Distribution Across Segments",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── Feature comparison bar charts ───────────────────────────────────

def plot_feature_comparison(summary_df: pd.DataFrame, feature: str, title: str) -> go.Figure:
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(summary_df))]
    fig = go.Figure(go.Bar(
        x=summary_df["SegmentLabel"],
        y=summary_df[feature],
        marker_color=colours,
        text=summary_df[feature].round(1),
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Segment",
        yaxis_title=feature,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── Scatter plot of two raw features ────────────────────────────────

def plot_scatter_2d(df: pd.DataFrame, x_col: str, y_col: str,
                    labels: np.ndarray, cluster_labels: dict) -> go.Figure:
    segment_names = [cluster_labels[l]["label"] for l in labels]
    plot_df = df[[x_col, y_col]].copy()
    plot_df["Segment"] = segment_names

    colour_map = _colour_map(labels, cluster_labels)

    fig = px.scatter(
        plot_df, x=x_col, y=y_col, color="Segment",
        color_discrete_map=colour_map,
        title=f"{y_col} vs {x_col} by Segment",
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig

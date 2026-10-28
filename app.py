"""
app.py – E-Commerce Customer Segmentation Dashboard
====================================================
Main Streamlit application.  Run with:
    streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

from src.data_generator import generate_dataset
from src.preprocessing import (
    validate_dataframe, clean_data, get_cleaning_report,
    scale_features, CLUSTERING_FEATURES,
)
from src.clustering import (
    compute_elbow, compute_silhouette_scores,
    train_kmeans, label_clusters, get_cluster_summary,
)
from src.visualization import (
    plot_elbow, plot_silhouette, plot_pca_clusters,
    plot_cluster_distribution, plot_feature_comparison,
    plot_scatter_2d,
)
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.15);
    }
    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-weight: 500;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0a2e 0%, #1a1145 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c7d2fe;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* Header badge */
    .header-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Sidebar – data source & parameters
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Customer Segmentation")
    st.markdown("---")

    st.markdown("### Data Source")
    data_source = st.radio(
        "Choose data source:",
        ("Generate Synthetic Data", "Upload CSV"),
        index=0,
    )

    df_raw: pd.DataFrame | None = None

    if data_source == "Generate Synthetic Data":
        n_customers = st.slider("Number of customers", 100, 2000, 500, 50)
        if st.button("🔄 Generate Dataset", use_container_width=True):
            st.session_state["df_raw"] = generate_dataset(n_customers)
        if "df_raw" not in st.session_state:
            st.session_state["df_raw"] = generate_dataset(500)
        df_raw = st.session_state["df_raw"]
        st.info("ℹ️ This is **synthetic demonstration data** generated for academic purposes.")
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                errors = validate_dataframe(df_raw)
                if errors:
                    for e in errors:
                        st.error(e)
                    df_raw = None
                else:
                    st.success(f"✅ Loaded {len(df_raw)} rows from uploaded file.")
                    st.session_state["df_raw"] = df_raw
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")
                df_raw = None
        elif "df_raw" in st.session_state and data_source == "Upload CSV":
            df_raw = st.session_state.get("df_raw")

    st.markdown("---")
    st.markdown("### Clustering Parameters")
    k = st.slider("Number of clusters (K)", 2, 10, 4)

# ─────────────────────────────────────────────────────────────────────
# Guard: need data
# ─────────────────────────────────────────────────────────────────────
if df_raw is None:
    st.markdown("# 🛒 E-Commerce Customer Segmentation")
    st.markdown("Upload a CSV or generate synthetic data from the sidebar to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Pipeline: clean → scale → cluster
# ─────────────────────────────────────────────────────────────────────
df_clean = clean_data(df_raw)
cleaning_report = get_cleaning_report(df_raw, df_clean)
X_scaled, scaler = scale_features(df_clean)

model = train_kmeans(X_scaled, k)
labels = model.labels_
sil_score = silhouette_score(X_scaled, labels)

cluster_labels = label_clusters(df_clean, labels, k)
summary_df = get_cluster_summary(df_clean, labels, cluster_labels)

# Attach segment info to the cleaned data
df_result = df_clean.copy()
df_result["Cluster"] = labels
df_result["Segment"] = [cluster_labels[l]["label"] for l in labels]

# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.markdown('<span class="header-badge">K-MEANS CLUSTERING</span>', unsafe_allow_html=True)
st.markdown("# 🛒 E-Commerce Customer Segmentation")
st.markdown(
    "Segment customers based on purchasing behaviour using **K-Means Clustering** "
    "to enable data-driven marketing strategies."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────
tab_overview, tab_segments, tab_elbow, tab_analysis, tab_explorer = st.tabs([
    "📊 Dashboard Overview",
    "🎯 Customer Segmentation",
    "📈 Elbow Method",
    "🔍 Cluster Analysis",
    "👤 Customer Explorer",
])

# ── A. Dashboard Overview ───────────────────────────────────────────
with tab_overview:
    st.markdown("## Dashboard Overview")

    # Find highest-spending segment
    top_segment = summary_df.loc[summary_df["AvgTotalSpending"].idxmax(), "SegmentLabel"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df_result):,}")
    col2.metric("Clusters (K)", k)
    col3.metric("Avg Spending", f"${df_result['TotalSpending'].mean():,.0f}")
    col4.metric("Top Segment", top_segment)

    st.markdown("")
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            plot_cluster_distribution(labels, cluster_labels),
            use_container_width=True,
            key="overview_distribution",
        )
    with col_right:
        st.plotly_chart(
            plot_pca_clusters(X_scaled, labels, cluster_labels),
            use_container_width=True,
            key="overview_pca",
        )

    st.markdown("---")
    st.markdown("### Data Cleaning Summary")
    c1, c2 = st.columns(2)
    c1.metric("Duplicates Removed", cleaning_report["duplicates_removed"])
    c2.metric("Rows After Cleaning", cleaning_report["rows_after_cleaning"])

    missing = cleaning_report["missing_values_before"]
    total_missing = sum(missing.values())
    if total_missing > 0:
        st.warning(f"⚠️ {total_missing} missing values were imputed with column medians.")
    else:
        st.success("✅ No missing values detected in clustering features.")

    st.metric("Silhouette Score", f"{sil_score:.4f}")
    st.caption(
        "Silhouette Score ranges from −1 to 1. Values closer to 1 indicate well-separated clusters."
    )

    # Download synthetic dataset
    st.markdown("---")
    st.markdown("### Download Dataset")
    csv_buf = io.StringIO()
    df_raw.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download Raw Dataset (CSV)",
        data=csv_buf.getvalue(),
        file_name="ecommerce_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── B. Customer Segmentation ────────────────────────────────────────
with tab_segments:
    st.markdown("## Customer Segmentation Visualization")

    st.plotly_chart(
        plot_pca_clusters(X_scaled, labels, cluster_labels),
        use_container_width=True,
        key="segmentation_pca",
    )
    st.caption(
        "**Note:** PCA is used only for visualisation. The K-Means model was trained "
        "on all standardised features, not on these two principal components."
    )

    st.markdown("### Feature-Space Scatter Plots")
    col_x, col_y = st.columns(2)
    with col_x:
        x_feat = st.selectbox("X-axis feature", CLUSTERING_FEATURES, index=1)
    with col_y:
        y_feat = st.selectbox("Y-axis feature", CLUSTERING_FEATURES, index=2)

    st.plotly_chart(
        plot_scatter_2d(df_clean, x_feat, y_feat, labels, cluster_labels),
        use_container_width=True,
        key="segmentation_scatter",
    )

    st.plotly_chart(
        plot_cluster_distribution(labels, cluster_labels),
        use_container_width=True,
        key="segmentation_distribution",
    )

# ── C. Elbow Method ─────────────────────────────────────────────────
with tab_elbow:
    st.markdown("## Elbow Method & Silhouette Analysis")
    st.markdown(
        "The **Elbow Method** plots the Within-Cluster Sum of Squares (inertia) for "
        "different values of K. The 'elbow' point — where inertia starts to decrease "
        "more slowly — suggests a good number of clusters."
    )

    elbow_data = compute_elbow(X_scaled)
    sil_data = compute_silhouette_scores(X_scaled)

    st.plotly_chart(plot_elbow(elbow_data), use_container_width=True, key="elbow_chart")

    st.markdown("---")
    st.markdown(
        "The **Silhouette Score** measures how similar each point is to its own cluster "
        "compared to other clusters. Higher is better (max = 1)."
    )
    st.plotly_chart(plot_silhouette(sil_data), use_container_width=True, key="silhouette_chart")

    best_k = sil_data["k_values"][int(np.argmax(sil_data["scores"]))]
    st.info(
        f"📌 Based on silhouette analysis, **K = {best_k}** yields the highest "
        f"silhouette score of **{max(sil_data['scores']):.4f}**."
    )

# ── D. Cluster Analysis ─────────────────────────────────────────────
with tab_analysis:
    st.markdown("## Cluster Analysis")

    st.markdown("### Segment Summary Table")
    display_cols = [
        "Cluster", "SegmentLabel", "CustomerCount",
        "AvgAnnualIncome", "AvgTotalSpending", "AvgNumPurchases",
        "AvgOrderValue", "AvgDaysSinceLastPurchase",
    ]
    st.dataframe(
        summary_df[display_cols].rename(columns={
            "SegmentLabel": "Segment",
            "CustomerCount": "Customers",
            "AvgAnnualIncome": "Avg Income ($)",
            "AvgTotalSpending": "Avg Spending ($)",
            "AvgNumPurchases": "Avg Purchases",
            "AvgOrderValue": "Avg Order ($)",
            "AvgDaysSinceLastPurchase": "Avg Days Since Purchase",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Feature Comparison Charts")
    chart_pairs = [
        ("AvgTotalSpending",         "Average Total Spending by Segment"),
        ("AvgAnnualIncome",          "Average Annual Income by Segment"),
        ("AvgNumPurchases",          "Average Number of Purchases by Segment"),
        ("AvgOrderValue",            "Average Order Value by Segment"),
        ("AvgDaysSinceLastPurchase", "Average Days Since Last Purchase by Segment"),
    ]
    for feat, title in chart_pairs:
        st.plotly_chart(
            plot_feature_comparison(summary_df, feat, title),
            use_container_width=True,
            key=f"comparison_{feat}",
        )

    st.markdown("---")
    st.markdown("### Business Recommendations")
    st.caption(
        "⚠️ Segment labels are **heuristic interpretations** of the clustering "
        "results and not ground-truth classifications."
    )
    for _, row in summary_df.iterrows():
        with st.expander(f"🏷️ {row['SegmentLabel']}  ({row['CustomerCount']} customers)"):
            st.markdown(f"**Recommendation:** {row['Recommendation']}")
            st.markdown(
                f"- Avg Income: **${row['AvgAnnualIncome']:,.0f}**\n"
                f"- Avg Spending: **${row['AvgTotalSpending']:,.0f}**\n"
                f"- Avg Purchases: **{row['AvgNumPurchases']:.0f}**\n"
                f"- Avg Order Value: **${row['AvgOrderValue']:,.0f}**\n"
                f"- Avg Days Since Purchase: **{row['AvgDaysSinceLastPurchase']:.0f}**"
            )

# ── E. Customer Explorer ────────────────────────────────────────────
with tab_explorer:
    st.markdown("## Customer Explorer")

    col_search, col_filter = st.columns(2)
    with col_search:
        search_id = st.text_input("🔍 Search by Customer ID", placeholder="e.g. CUST-0042")
    with col_filter:
        segment_options = ["All Segments"] + sorted(
            [cluster_labels[c]["label"] for c in cluster_labels]
        )
        selected_segment = st.selectbox("Filter by Segment", segment_options)

    filtered = df_result.copy()
    if search_id:
        filtered = filtered[filtered["CustomerID"].str.contains(search_id, case=False, na=False)]
    if selected_segment != "All Segments":
        filtered = filtered[filtered["Segment"] == selected_segment]

    st.markdown(f"**Showing {len(filtered)} customer(s)**")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if search_id and len(filtered) == 1:
        cust = filtered.iloc[0]
        st.markdown("### Customer Details")
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Segment", cust["Segment"])
        dc2.metric("Annual Income", f"${cust['AnnualIncome']:,.0f}")
        dc3.metric("Total Spending", f"${cust['TotalSpending']:,.0f}")

    st.markdown("---")
    csv_result = io.StringIO()
    df_result.to_csv(csv_result, index=False)
    st.download_button(
        "⬇️ Download Segmented Dataset (CSV)",
        data=csv_result.getvalue(),
        file_name="segmented_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:0.85rem;'>"
    "E-Commerce Customer Segmentation — K-Means Clustering Project &nbsp;|&nbsp; "
    "Built with Streamlit, Scikit-learn & Plotly"
    "</div>",
    unsafe_allow_html=True,
)

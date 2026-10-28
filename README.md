# E-Commerce Customer Segmentation Using K-Means Clustering

<p align="center">
  <strong>🛒 A Machine Learning Project for Customer Segmentation</strong><br>
  Built with Python · Scikit-learn · Streamlit · Plotly
</p>

---

## Abstract

This project implements an **E-Commerce Customer Segmentation** system using the **K-Means Clustering** algorithm. The application analyses customer purchasing behaviour — including spending patterns, purchase frequency, income levels, and engagement metrics — to automatically group customers into distinct segments. A professional Streamlit dashboard provides interactive visualisations, cluster analysis, and actionable business recommendations. The project demonstrates the practical application of unsupervised machine learning in the e-commerce domain.

---

## 1. Introduction

Customer segmentation is the process of dividing customers into groups based on common characteristics. In the e-commerce industry, understanding different customer groups enables businesses to design targeted marketing campaigns, personalise user experiences, and optimise resource allocation.

This project uses **K-Means Clustering**, an unsupervised machine-learning algorithm, to discover natural groupings within customer data without relying on pre-labelled categories.

---

## 2. Problem Statement

E-commerce businesses accumulate vast amounts of customer transaction data but often lack the tools to extract actionable insights. Without segmentation, companies apply one-size-fits-all marketing strategies that fail to resonate with diverse customer groups. This leads to wasted marketing budgets, lower conversion rates, and increased customer churn.

**The goal of this project is to build an automated customer segmentation system that:**
- Analyses customer purchasing behaviour from transactional data.
- Groups customers into meaningful segments using K-Means Clustering.
- Presents the results through an interactive dashboard with business recommendations.

---

## 3. Project Objectives

1. Generate (or accept uploaded) e-commerce customer data with realistic purchasing patterns.
2. Preprocess and standardise the data for clustering.
3. Determine the optimal number of clusters using the Elbow Method and Silhouette Analysis.
4. Train a K-Means model and assign each customer to a segment.
5. Interpret clusters with human-readable labels and actionable recommendations.
6. Present all results in a professional Streamlit dashboard.

---

## 4. Existing System vs Proposed System

### Existing System
| Aspect | Description |
|--------|-------------|
| Approach | Manual segmentation based on business rules (e.g., spending thresholds) |
| Scalability | Difficult to scale as the number of features grows |
| Objectivity | Subject to human bias and arbitrary cut-offs |
| Adaptability | Requires manual re-definition when customer patterns shift |

### Proposed System
| Aspect | Description |
|--------|-------------|
| Approach | Automated, data-driven segmentation using K-Means Clustering |
| Scalability | Handles multiple features simultaneously |
| Objectivity | Groups are determined by the algorithm based on data patterns |
| Adaptability | Re-training the model automatically captures new patterns |
| Visualisation | Interactive dashboard with charts, filters, and export options |

---

## 5. Dataset Description

The project uses a **synthetic dataset** of 500 customers (configurable up to 2,000) with the following features:

| Column | Description | Type |
|--------|-------------|------|
| `CustomerID` | Unique customer identifier | String |
| `Gender` | Male / Female | Categorical |
| `Age` | Customer age (18–75) | Integer |
| `AnnualIncome` | Yearly income in USD | Integer |
| `TotalSpending` | Total amount spent on the platform | Float |
| `NumPurchases` | Number of completed purchases | Integer |
| `AvgOrderValue` | Average value per order (USD) | Float |
| `WebsiteVisitsPerMonth` | Monthly site visits | Integer |
| `DaysSinceLastPurchase` | Recency — days since the last order | Integer |

The synthetic data is generated with five latent archetypes (high-value, loyal, budget, occasional, at-risk) plus Gaussian noise so that the K-Means algorithm can discover meaningful — but not trivially separable — clusters.

Users can also **upload their own CSV** dataset. The application validates that required columns are present and displays helpful error messages if they are not.

---

## 6. Software & Hardware Requirements

### Software
| Component | Requirement |
|-----------|-------------|
| Operating System | Windows 10/11 |
| Python | 3.10 or later |
| Key Libraries | streamlit, pandas, numpy, scikit-learn, plotly, matplotlib, seaborn |
| Browser | Any modern browser (Chrome, Edge, Firefox) |

### Hardware
| Component | Minimum |
|-----------|---------|
| Processor | Intel i3 or equivalent |
| RAM | 4 GB |
| Disk Space | 500 MB |

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│  (Dashboard Overview │ Segmentation │ Elbow │   │
│   Cluster Analysis │ Customer Explorer)         │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │    Application Layer    │
        │       (app.py)          │
        └────────────┬────────────┘
                     │
   ┌─────────────────┼─────────────────┐
   │                 │                 │
┌──▼───┐      ┌──────▼──────┐   ┌─────▼──────┐
│ Data │      │Preprocessing│   │ Clustering │
│Generator│   │  & Scaling  │   │  (K-Means) │
│      │      │             │   │            │
└──────┘      └─────────────┘   └────────────┘
                                       │
                                ┌──────▼──────┐
                                │Visualization│
                                │  (Plotly)   │
                                └─────────────┘
```

---

## 8. Methodology

### Step-by-step Pipeline

1. **Data Collection / Generation** — Generate synthetic data or load a user-uploaded CSV.
2. **Data Cleaning** — Remove duplicates; impute missing numerical values with column medians.
3. **Feature Selection** — Select the seven numerical features listed in the dataset description. Exclude `CustomerID` and `Gender` from clustering.
4. **Feature Scaling** — Standardise features using `StandardScaler` (zero mean, unit variance).
5. **Optimal K Selection** — Use the Elbow Method and Silhouette Analysis.
6. **Model Training** — Train K-Means with the selected K.
7. **Cluster Assignment** — Assign each customer a cluster label.
8. **Interpretation** — Map clusters to business-friendly segment names using a composite scoring heuristic.
9. **Presentation** — Display results in the Streamlit dashboard.

---

## 9. K-Means Clustering — Explained

### Algorithm
K-Means is an **unsupervised** machine-learning algorithm that partitions data into **K** clusters by minimising the **Within-Cluster Sum of Squares (WCSS / inertia)**.

### Steps
1. **Initialise** K centroids (Scikit-learn uses the k-means++ method).
2. **Assign** each data point to the nearest centroid.
3. **Update** each centroid to the mean of its assigned points.
4. **Repeat** steps 2–3 until centroids stabilise or the maximum number of iterations is reached.

### Parameters Used
| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_clusters` | User-selected (default 4) | Configurable via sidebar slider |
| `random_state` | 42 | Ensures reproducibility |
| `n_init` | 10 | Runs the algorithm 10 times with different seeds and keeps the best result |

### Why K-Means?
- Simple to implement and explain.
- Efficient on moderately sized datasets.
- Well-suited for spherical, roughly equally-sized clusters.

---

## 10. StandardScaler — Explained

`StandardScaler` transforms each feature so that it has a **mean of 0** and a **standard deviation of 1**.

**Formula:**

```
z = (x − μ) / σ
```

Where `x` is the original value, `μ` is the feature mean, and `σ` is the feature standard deviation.

**Why it matters:**
K-Means uses Euclidean distance. Without scaling, features with large ranges (e.g., `AnnualIncome` in tens of thousands) would dominate the distance calculation, drowning out features like `NumPurchases`.

---

## 11. Elbow Method — Explained

The Elbow Method plots **inertia** (WCSS) against the number of clusters K.

- As K increases, inertia decreases (more clusters ⇒ tighter grouping).
- The rate of decrease slows at a certain K — the **"elbow point"**.
- The elbow suggests a K that balances model complexity with goodness of fit.

> The dashboard provides an interactive elbow plot for K = 2 to K = 10.

---

## 12. Silhouette Score — Explained

The Silhouette Score measures clustering quality on a scale of **−1 to +1**.

| Range | Interpretation |
|-------|----------------|
| Close to +1 | Data points are well-matched to their own cluster and poorly matched to neighbours |
| Close to 0 | Data points are on or near the boundary between clusters |
| Negative | Data points may have been assigned to the wrong cluster |

**Formula (per sample):**

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

Where `a(i)` is the average intra-cluster distance and `b(i)` is the average nearest-cluster distance.

---

## 13. Results & Visualisations

The dashboard presents:

1. **Dashboard Overview** — KPI metrics (total customers, clusters, average spending, top segment), distribution donut chart, PCA scatter plot, and a data-cleaning summary.
2. **Customer Segmentation** — PCA 2-D projection, interactive feature-space scatter plots with user-selectable axes, and segment distribution.
3. **Elbow Method** — Inertia vs K plot and silhouette bar chart with the recommended K highlighted.
4. **Cluster Analysis** — Summary table of per-segment statistics, bar charts comparing features across segments, and expandable business recommendations.
5. **Customer Explorer** — Search by ID, filter by segment, view individual customer details, and download the segmented dataset as CSV.

> **Note:** K-Means is an unsupervised algorithm. There is no "accuracy" metric as in classification. We evaluate with **inertia** and **silhouette score**.

---

## 14. Advantages

- Fully automated, data-driven segmentation.
- Works offline — no API keys or paid services required.
- Interactive dashboard for intuitive exploration.
- Modular codebase that is easy to extend.
- Supports user-uploaded datasets.
- Reproducible results via fixed random state.

## 15. Limitations

- K-Means assumes spherical clusters of roughly equal size.
- The number of clusters (K) must be pre-specified.
- Sensitive to outliers and the initial centroid placement.
- Segment labels are heuristic interpretations, not ground-truth classifications.
- Categorical features (`Gender`) are not included in clustering.

---

## 16. Future Scope

1. Incorporate additional clustering algorithms (DBSCAN, Hierarchical Clustering) for comparison.
2. Apply dimensionality reduction techniques such as t-SNE for richer visualisation.
3. Integrate with a real database or CRM system for live customer data.
4. Add predictive analytics — predict which segment a new customer will fall into.
5. Implement automated A/B testing of marketing strategies per segment.
6. Deploy the dashboard on a cloud platform (e.g., Streamlit Cloud, AWS).

---

## 17. Conclusion

This project successfully demonstrates how **K-Means Clustering** can be applied to e-commerce customer data to uncover meaningful customer segments. The interactive Streamlit dashboard makes the results accessible to non-technical stakeholders, while the modular Python codebase ensures maintainability and extensibility. The Elbow Method and Silhouette Analysis provide objective guidance for selecting the number of clusters, and the heuristic labelling system translates raw cluster assignments into actionable business insights.

---

## 18. Installation & Execution

### Prerequisites
- **Python 3.10+** installed and available on `PATH`.
- **pip** (bundled with Python).

### Steps

```bash
# 1. Clone or navigate to the project directory
cd ecommerce-customer-segmentation

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows:
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the dashboard
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## 19. Project Structure

```
ecommerce-customer-segmentation/
├── app.py                  # Streamlit dashboard (main entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
├── data/                   # (generated CSVs are saved here)
└── src/
    ├── __init__.py
    ├── data_generator.py   # Synthetic dataset generation
    ├── preprocessing.py    # Data cleaning & feature scaling
    ├── clustering.py       # K-Means training & segment labelling
    └── visualization.py    # Plotly chart builders
```

---

## 20. Viva Preparation — Questions & Answers

### Q1: What is K-Means Clustering?
**A:** K-Means is an unsupervised machine-learning algorithm that partitions data into K groups by minimising the Within-Cluster Sum of Squares (inertia). It iteratively assigns points to the nearest centroid and updates centroids until convergence.

### Q2: Why did you choose K-Means over other clustering algorithms?
**A:** K-Means is simple to implement, computationally efficient, and easy to interpret — making it ideal for this academic project. It works well when clusters are roughly spherical and similarly sized.

### Q3: How do you decide the value of K?
**A:** We use two techniques:
- **Elbow Method** — We plot inertia vs K and look for the "elbow" where the rate of decrease slows.
- **Silhouette Analysis** — We pick the K with the highest silhouette score, indicating well-separated clusters.

### Q4: Why is StandardScaler necessary?
**A:** K-Means uses Euclidean distance. Features like `AnnualIncome` (range: 10,000–200,000) would dominate distance computations over features like `NumPurchases` (range: 1–150). StandardScaler normalises all features to the same scale (mean = 0, std = 1).

### Q5: What is the Silhouette Score?
**A:** It measures how similar a point is to its own cluster vs the nearest neighbouring cluster, on a scale from −1 to +1. Higher values indicate better-defined clusters.

### Q6: Is K-Means supervised or unsupervised?
**A:** Unsupervised. It does not require labelled data. The algorithm discovers groupings based on the inherent structure of the data.

### Q7: What is PCA and why did you use it?
**A:** PCA (Principal Component Analysis) reduces multi-dimensional data to two dimensions for visualisation. We use it only for plotting — the K-Means model is trained on all seven standardised features, not on the PCA components.

### Q8: Can you measure accuracy for K-Means?
**A:** No. Accuracy is a supervised metric that requires ground-truth labels. For unsupervised clustering, we use inertia (lower is better for a given K) and silhouette score (higher is better).

### Q9: What are the limitations of K-Means?
**A:** (1) Requires specifying K in advance. (2) Assumes spherical clusters of similar size. (3) Sensitive to outliers. (4) May converge to a local minimum (mitigated by `n_init=10`).

### Q10: How are the segment labels assigned?
**A:** A heuristic scoring system ranks clusters by a composite of spending, engagement, and income statistics. The highest-scoring cluster is labelled "High-Value Customers", the next "Loyal Customers", and so on. Labels are interpretive, not absolute classifications.

### Q11: What libraries did you use?
**A:** Pandas and NumPy for data handling, Scikit-learn for K-Means and StandardScaler, Plotly for interactive charts, and Streamlit for the web dashboard.

### Q12: Can the system handle real-world data?
**A:** Yes. Users can upload their own CSV with the required columns, and the application will validate, clean, and segment the data automatically.

---

*© 2026 — E-Commerce Customer Segmentation Project*

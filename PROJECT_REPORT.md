# Academic Project Report & Reference Document

## Project Title
**E-Commerce Customer Segmentation Using K-Means Clustering Algorithm**

---

## 1. Executive Summary & Abstract
In modern e-commerce, businesses accumulate vast amounts of transactional and engagement data. A one-size-fits-all marketing approach often results in customer churn, low conversion rates, and inefficient marketing spend.

This project implements an automated, unsupervised machine learning framework utilizing **K-Means Clustering** to segment customers based on behavioral patterns. The system evaluates 7 core numeric attributes:
- Annual Income
- Total Spending
- Number of Purchases
- Average Order Value (AOV)
- Website Visits per Month
- Recency (Days Since Last Purchase)
- Age

Features are scaled using **StandardScaler** ($\mu=0, \sigma=1$). Optimal cluster counts are determined using the **Elbow Method (Inertia/WCSS)** and **Silhouette Analysis**. 

An interactive **Streamlit web dashboard** provides:
- Executive KPI overview
- 2D Principal Component Analysis (PCA) projection
- Interactive feature-space scatter plots with dynamic axis selection
- Quantitative cluster profiling with business recommendations
- An interactive customer explorer with real-time filtering and CSV export

---

## 2. Problem Statement & Motivation
Legacy e-commerce segmentation relies heavily on manual rules and arbitrary thresholds (e.g., customers spending > $5,000 are labeled "VIP"). This approach has significant drawbacks:
1. **Curse of Dimensionality:** Humans cannot effectively evaluate multi-dimensional trade-offs (e.g., high recency + low spending vs. high visits + high order value).
2. **Subjectivity & Bias:** Fixed cutoffs do not adapt to shifts in market dynamics or organic customer behavior.
3. **Lack of Granularity:** Nuanced micro-segments (e.g., high-income occasional buyers vs. frequent budget buyers) are completely missed.

**Proposed Solution:** An unsupervised learning pipeline using K-Means clustering to partition customer records into cohesive, mathematically optimal clusters based on normalized behavioral distance metrics in multi-dimensional feature space.

---

## 3. Technology Stack & Architecture

### Technology Stack
- **Programming Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-learn (`KMeans`, `StandardScaler`, `silhouette_score`, `PCA`)
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Web Interface:** Streamlit (custom dark-theme layout)

### Project Directory Structure
```
ecommerce-customer-segmentation/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive repository documentation & Viva guide
├── PROJECT_REPORT.md           # Full academic report and reference guide
├── .gitignore                  # Git ignore rules
├── data/
│   └── ecommerce_customers.csv # Demonstration dataset (500 records)
└── src/
    ├── __init__.py             # Python package marker
    ├── data_generator.py       # Synthetic dataset generation engine
    ├── preprocessing.py        # Data hygiene, median imputation & StandardScaler
    ├── clustering.py           # K-Means training, Elbow/Silhouette analysis & heuristic labeling
    └── visualization.py        # Dark-mode Plotly chart builders
```

---

## 4. System Workflow & Methodology

```
[Raw Data: Synthetic Engine OR User Uploaded CSV]
                         │
                         ▼
[Data Preprocessing & Hygiene]
  - Check missing values & impute with column median
  - Remove duplicate rows
  - Filter 7 numerical features (exclude CustomerID, Gender)
                         │
                         ▼
[Feature Standardization]
  - StandardScaler: z = (x - mean) / std
                         │
                         ▼
[Hyperparameter Optimization]
  - Elbow Method: Within-Cluster Sum of Squares (WCSS vs K)
  - Silhouette Analysis (K=2 to 10)
                         │
                         ▼
[K-Means Model Training]
  - Scikit-learn KMeans (random_state=42, n_init=10, k-means++)
  - Cluster label assignment
                         │
                         ▼
[Cluster Profiling & Heuristic Labeling]
  - Group-by averages calculated on original unscaled values
  - Dynamic composite ranking (Spending + Engagement + Income)
  - Business recommendations per segment
                         │
                         ▼
[Interactive Streamlit Dashboard]
  - Tab 1: Executive Overview & KPI Metrics
  - Tab 2: 2D PCA & Feature-Space Visualizations
  - Tab 3: Elbow Curve & Silhouette Bar Chart
  - Tab 4: Cluster Statistical Breakdown & Actionable Strategies
  - Tab 5: Customer Explorer, ID Search & CSV Export
```

---

## 5. Mathematical & Algorithmic Foundations

### 5.1 K-Means Clustering
K-Means partitions $n$ data points into $K$ clusters by minimizing the Within-Cluster Sum of Squares (Inertia):

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in S_k} ||x_i - \mu_k||^2$$

Where:
- $\mu_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} x_i$ is the centroid of cluster $k$.
- $||x_i - \mu_k||^2$ is the squared Euclidean distance between observation $x_i$ and centroid $\mu_k$.

**Key Parameters Used:**
- `n_clusters`: Configured dynamically via user slider (default $K=4$).
- `init='k-means++'`: Accelerates convergence and avoids poor local minima.
- `n_init=10`: Runs the algorithm 10 times with different centroid seeds and picks the lowest inertia model.
- `random_state=42`: Ensures complete reproducibility across runs.

### 5.2 StandardScaler Normalization
Because Euclidean distance is calculated across all dimensions:

$$d(p, q) = \sqrt{\sum_{j=1}^{d} (p_j - q_j)^2}$$

features with large numerical magnitudes (e.g. `AnnualIncome` in tens of thousands) would dwarf features with small magnitudes (e.g. `NumPurchases` in units). 

StandardScaler scales each feature independently:

$$z = \frac{x - \mu}{\sigma}$$

giving every feature a mean of 0 and a standard deviation of 1.

### 5.3 Optimal Cluster Validation

#### A. Elbow Method
Inertia decreases monotonically as $K$ increases. The "elbow" point indicates where the marginal improvement drops sharply, balancing cluster compactness with model simplicity.

#### B. Silhouette Score
Measures cohesion versus separation for each sample:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = mean intra-cluster distance.
- $b(i)$ = mean nearest-cluster distance.
- A score near $+1.0$ indicates well-separated clusters; scores near $0$ indicate boundary overlap.

### 5.4 Principal Component Analysis (PCA)
Used **solely for 2D visualization**. PCA computes the principal eigenvectors of the feature covariance matrix, projecting 7-dimensional standardized data onto the two axes of highest variance ($PC_1, PC_2$) without altering the original K-Means model.

---

## 6. Dataset Specifications

The dataset uses 7 behavioral metrics and 2 metadata columns:

| Column Name | Data Type | Value Range | Business Meaning |
|---|---|---|---|
| `CustomerID` | String | `CUST-0001` to `CUST-N` | Unique primary key (excluded from clustering) |
| `Gender` | Categorical | Male / Female | Demographic identifier (excluded from clustering) |
| `Age` | Integer | 18 – 75 | Customer demographic life-stage |
| `AnnualIncome` | Integer | \$10,000 – \$200,000 | Purchasing power |
| `TotalSpending` | Float | \$50 – \$30,000 | Cumulative monetary contribution |
| `NumPurchases` | Integer | 1 – 150 | Purchase frequency |
| `AvgOrderValue` | Float | \$10 – \$500 | Average basket value |
| `WebsiteVisitsPerMonth` | Integer | 1 – 50 | Digital browsing engagement |
| `DaysSinceLastPurchase` | Integer | 1 – 365 | Recency / Churn risk |

---

## 7. Cluster Profiles & Business Strategies ($K=4$)

| Segment Label | Profile Characteristics | Business Strategy & Recommendation |
|---|---|---|
| **High-Value Customers** | Top income, highest annual spend, high purchase frequency, low recency (< 10 days) | VIP concierge, exclusive early product drops, loyalty rewards, premium personalized service. |
| **Loyal Customers** | Moderate-to-high income, steady purchase frequency, strong website engagement | Tiered cashback, subscription perks, referral bonuses, retention campaigns. |
| **Potential Customers** | Lower-to-moderate income, moderate visits, lower basket size | Limited-time deals, bundle discounts, Buy-Now-Pay-Later (BNPL) options, cart nudges. |
| **At-Risk Customers** | High recency (> 90 days), declining visit frequency, low spending | Win-back email series, steep reactivation coupons, customer feedback surveys. |

---

## 8. Viva Examination Preparation: Top Q&A

**Q1: Why is K-Means called an unsupervised algorithm?**  
*Answer:* Because it groups data based entirely on inherent patterns and geometric distances without using pre-existing target labels.

**Q2: Why not evaluate K-Means with Accuracy or F1-Score?**  
*Answer:* Accuracy requires ground-truth labels. In unsupervised clustering, evaluation is performed using internal validation metrics such as **Inertia (WCSS)** and the **Silhouette Coefficient**.

**Q3: What role does StandardScaler play in K-Means?**  
*Answer:* K-Means relies on Euclidean distance. Features with large numerical ranges (like Income in \$10,000s) would dominate smaller features (like Visits in single digits). StandardScaler normalizes all dimensions to zero mean and unit variance.

**Q4: Why is PCA used in this project?**  
*Answer:* To visualize a 7-dimensional clustering space on a 2-dimensional plane. PCA projects data onto its top two principal components for display purposes only; K-Means is trained on all 7 standardized features.

**Q5: What are the main limitations of K-Means?**  
*Answer:*
1. $K$ must be predefined.
2. Assumes spherical, equally-sized clusters.
3. Sensitive to outliers.
4. Can converge to local minima (mitigated using `k-means++` and `n_init=10`).

---

## 9. Verification & Local Execution

To run the application locally on Windows:

```powershell
# 1. Open PowerShell and navigate to the project directory
cd "c:\Users\Parth\E-commerse segmentation"

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Launch the dashboard
streamlit run app.py
```

The application runs at: **http://localhost:8501**

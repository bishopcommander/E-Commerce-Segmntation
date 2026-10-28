"""
preprocessing.py
----------------
Data-cleaning and feature-scaling utilities for the customer segmentation
pipeline.  All transformations are deterministic given the same input.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Features used for clustering (excludes identifiers and categorical cols)
CLUSTERING_FEATURES = [
    "Age",
    "AnnualIncome",
    "TotalSpending",
    "NumPurchases",
    "AvgOrderValue",
    "WebsiteVisitsPerMonth",
    "DaysSinceLastPurchase",
]


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error messages (empty ⇒ valid)."""
    errors: list[str] = []
    required = set(CLUSTERING_FEATURES)
    present = set(df.columns)
    missing = required - present
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
    # Check at least some rows
    if len(df) < 10:
        errors.append("Dataset must contain at least 10 rows.")
    return errors


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicates.

    - Duplicate rows are dropped.
    - Missing numerical values in clustering features are filled with column
      medians.
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    for col in CLUSTERING_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def get_cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """Return a summary of cleaning actions taken."""
    missing_before = df_raw[CLUSTERING_FEATURES].isna().sum().to_dict()
    duplicates_removed = len(df_raw) - len(df_clean)
    return {
        "missing_values_before": missing_before,
        "duplicates_removed": duplicates_removed,
        "rows_after_cleaning": len(df_clean),
    }


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardize clustering features (zero mean, unit variance).

    Returns
    -------
    X_scaled : np.ndarray   – shape (n_samples, n_features)
    scaler   : StandardScaler – fitted scaler (kept for inverse-transform)
    """
    X = df[CLUSTERING_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

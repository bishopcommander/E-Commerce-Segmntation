"""
data_generator.py
-----------------
Generates a synthetic e-commerce customer dataset with realistic purchasing
patterns so that K-Means can discover meaningful segments.

Five latent archetypes are seeded (high-value, loyal, budget, occasional,
at-risk) with Gaussian noise added to each feature so that clusters overlap
naturally and the algorithm has genuine work to do.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def generate_dataset(n_customers: int = 500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a DataFrame with *n_customers* rows of synthetic e-commerce data.

    Parameters
    ----------
    n_customers : int
        Total number of customer rows to generate (default 500).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: CustomerID, Gender, Age, AnnualIncome, TotalSpending,
        NumPurchases, AvgOrderValue, WebsiteVisitsPerMonth,
        DaysSinceLastPurchase.
    """
    rng = np.random.RandomState(random_state)

    # ── archetype definitions (mean values) ──────────────────────────
    # Each row: [Age, AnnualIncome, TotalSpending, NumPurchases,
    #            AvgOrderValue, WebsiteVisitsPerMonth, DaysSinceLastPurchase]
    archetypes = {
        "high_value":   [42, 95_000, 8_500, 55, 155, 22, 8],
        "loyal":        [35, 60_000, 4_200, 40, 105, 18, 15],
        "budget":       [28, 30_000, 1_200, 20,  60, 12, 30],
        "occasional":   [50, 55_000, 1_800, 10, 180,  5, 60],
        "at_risk":      [38, 45_000,   600,  5, 120,  3, 120],
    }

    # Noise standard-deviations (same order as above)
    noise_std = [8, 15_000, 1_500, 12, 35, 5, 15]

    # Proportions per archetype
    proportions = [0.15, 0.25, 0.25, 0.20, 0.15]

    segment_sizes = _split_sizes(n_customers, proportions)

    rows = []
    for (_, means), size in zip(archetypes.items(), segment_sizes):
        for _ in range(size):
            row = [m + rng.randn() * s for m, s in zip(means, noise_std)]
            rows.append(row)

    data = np.array(rows)

    # Clip to realistic ranges
    data[:, 0] = np.clip(data[:, 0], 18, 75)            # Age
    data[:, 1] = np.clip(data[:, 1], 10_000, 200_000)   # AnnualIncome
    data[:, 2] = np.clip(data[:, 2], 50, 30_000)        # TotalSpending
    data[:, 3] = np.clip(data[:, 3], 1, 150)             # NumPurchases
    data[:, 4] = np.clip(data[:, 4], 10, 500)            # AvgOrderValue
    data[:, 5] = np.clip(data[:, 5], 1, 50)              # WebsiteVisitsPerMonth
    data[:, 6] = np.clip(data[:, 6], 1, 365)             # DaysSinceLastPurchase

    # Shuffle rows so archetypes aren't contiguous
    rng.shuffle(data)

    df = pd.DataFrame(data, columns=[
        "Age", "AnnualIncome", "TotalSpending", "NumPurchases",
        "AvgOrderValue", "WebsiteVisitsPerMonth", "DaysSinceLastPurchase",
    ])

    # Round to sensible precision
    df["Age"] = df["Age"].astype(int)
    df["AnnualIncome"] = df["AnnualIncome"].round(-2).astype(int)
    df["TotalSpending"] = df["TotalSpending"].round(2)
    df["NumPurchases"] = df["NumPurchases"].astype(int)
    df["AvgOrderValue"] = df["AvgOrderValue"].round(2)
    df["WebsiteVisitsPerMonth"] = df["WebsiteVisitsPerMonth"].astype(int)
    df["DaysSinceLastPurchase"] = df["DaysSinceLastPurchase"].astype(int)

    # Gender column (random)
    df.insert(0, "Gender", rng.choice(["Male", "Female"], size=len(df)))

    # CustomerID
    df.insert(0, "CustomerID", [f"CUST-{i+1:04d}" for i in range(len(df))])

    return df


# ── helper ───────────────────────────────────────────────────────────
def _split_sizes(total: int, proportions: list[float]) -> list[int]:
    """Split *total* into integer sizes matching *proportions*."""
    sizes = [int(total * p) for p in proportions]
    remainder = total - sum(sizes)
    for i in range(remainder):
        sizes[i % len(sizes)] += 1
    return sizes

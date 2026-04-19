"""
workflow/segmentation.py
------------------------
Stages 7 and 8 of the CheckPoint3 flow: segment counties into pricing tiers.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PREMIUM_MULTIPLIERS = {"LOW": 0.85, "MEDIUM": 1.00, "HIGH": 1.25}
TIER_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
DEFAULT_CLUSTER_COLUMNS = [
    "predicted_rate",
    "opioid_rate_lag",
    "opioid_share",
    "la_rate",
    "rate_1y_chg",
]


def _get_cluster_columns(df: pd.DataFrame, cluster_columns: list[str] | None) -> list[str]:
    if cluster_columns is not None:
        return cluster_columns
    available = [column for column in DEFAULT_CLUSTER_COLUMNS if column in df.columns]
    return available if available else ["predicted_rate"]


def find_optimal_k(
    county_predictions: pd.DataFrame,
    k_range: range = range(2, 9),
    cluster_columns: list[str] | None = None,
) -> list[tuple[int, float]]:
    """Return inertias for the elbow analysis used to justify k=3."""
    columns = _get_cluster_columns(county_predictions, cluster_columns)
    X = StandardScaler().fit_transform(county_predictions[columns].fillna(0))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append((k, float(km.inertia_)))
    return inertias


def assign_risk_tiers(
    county_predictions: pd.DataFrame,
    cluster_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Cluster counties into Low, Medium, and High premium tiers."""
    if "predicted_rate" not in county_predictions.columns:
        raise ValueError("county_predictions must include a 'predicted_rate' column.")

    df = county_predictions.copy()
    columns = _get_cluster_columns(df, cluster_columns)
    X = StandardScaler().fit_transform(df[columns].fillna(0))

    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    raw_clusters = km.fit_predict(X)

    cluster_means = (
        pd.DataFrame({"cluster": raw_clusters, "predicted_rate": df["predicted_rate"]})
        .groupby("cluster")["predicted_rate"]
        .mean()
        .sort_values()
    )
    remap = {cluster_id: rank for rank, cluster_id in enumerate(cluster_means.index)}

    df["cluster_raw"] = [remap[cluster] for cluster in raw_clusters]
    df["risk_tier"] = df["cluster_raw"].map(TIER_LABELS)
    df["premium_multiplier"] = df["risk_tier"].map(PREMIUM_MULTIPLIERS)
    return df


def summarize_tiers(tiered_df: pd.DataFrame) -> pd.DataFrame:
    """Return the county counts and average rates per risk tier."""
    aggregations = {
        "avg_predicted_rate": ("predicted_rate", "mean"),
        "premium_multiplier": ("premium_multiplier", "first"),
    }
    if "county_fips" in tiered_df.columns:
        aggregations["counties"] = ("county_fips", "count")
    else:
        aggregations["counties"] = ("risk_tier", "size")
    if "actual_rate" in tiered_df.columns:
        aggregations["avg_actual_rate"] = ("actual_rate", "mean")

    return tiered_df.groupby("risk_tier").agg(**aggregations).reindex(["LOW", "MEDIUM", "HIGH"])


def get_top_high_risk(tiered_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top predicted counties in the High tier."""
    columns = [
        column
        for column in [
            "county_name",
            "state",
            "predicted_rate",
            "actual_rate",
            "premium_multiplier",
        ]
        if column in tiered_df.columns
    ]
    return (
        tiered_df[tiered_df["risk_tier"] == "HIGH"]
        .sort_values("predicted_rate", ascending=False)
        .head(n)[columns]
    )

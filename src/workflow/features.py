"""
workflow/features.py
--------------------
Stage 3 of the CheckPoint3 flow: engineer the CP3 momentum and intensity features.
"""

from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the engineered CP3 features used by the final model."""
    df = df.copy()
    df["opioid_lag_sq"] = df["opioid_rate_lag"] ** 2
    df["opioid_share"] = df["opioid_claims"] / (df["total_claims"] + 1)
    df["la_share"] = df["la_rate"] / (df["opioid_rate_lag"] + 0.01)
    df["prescriber_ratio"] = df["opioid_prescribers"] / (df["prescribers"] + 1)

    if "rate_1y_chg" not in df.columns:
        df["rate_1y_chg"] = 0.0
    if "rate_5y_chg" not in df.columns:
        df["rate_5y_chg"] = 0.0

    return df


def get_feature_columns() -> list[str]:
    """Return the canonical CP3 feature list."""
    return [
        "opioid_rate_lag",
        "opioid_lag_sq",
        "opioid_share",
        "la_share",
        "prescriber_ratio",
        "rate_1y_chg",
        "rate_5y_chg",
    ]


def validate_features(df: pd.DataFrame) -> None:
    """Check that the raw inputs needed for feature engineering are present."""
    required_cols = [
        "opioid_rate_lag",
        "opioid_claims",
        "total_claims",
        "la_rate",
        "opioid_prescribers",
        "prescribers",
    ]
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}"
        )

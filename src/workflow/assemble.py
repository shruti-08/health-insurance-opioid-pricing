"""
workflow/assemble.py
--------------------
Stage 2 of the CheckPoint3 flow: build train/test frames with lag fields and
apply the CP2/CP3 outlier cutoff.
"""

from __future__ import annotations

import pandas as pd


def build_model_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outlier_sigma: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build notebook-style train/test datasets with lag columns from the train year."""
    required = {"county_fips", "opioid_rate", "la_rate"}
    missing_train = sorted(required - set(train_df.columns))
    missing_test = sorted(required - set(test_df.columns))
    if missing_train or missing_test:
        raise ValueError(
            "Missing required columns for dataset assembly. "
            f"train missing={missing_train}, test missing={missing_test}"
        )

    lag_df = train_df[["county_fips", "opioid_rate", "la_rate"]].rename(
        columns={"opioid_rate": "opioid_rate_lag", "la_rate": "la_rate_lag"}
    )

    train_prepared = train_df.merge(lag_df, on="county_fips", how="inner")
    test_prepared = test_df.merge(lag_df, on="county_fips", how="inner")

    cutoff = train_prepared["opioid_rate"].mean() + (
        outlier_sigma * train_prepared["opioid_rate"].std()
    )
    train_prepared = train_prepared[train_prepared["opioid_rate"] <= cutoff].copy()
    test_prepared = test_prepared[test_prepared["opioid_rate"] <= cutoff].copy()
    return train_prepared, test_prepared, float(cutoff)

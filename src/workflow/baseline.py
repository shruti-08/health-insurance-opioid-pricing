"""
workflow/baseline.py
--------------------
Stage 4 of the CheckPoint3 flow: evaluate the CP2-style linear regression baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


BASELINE_FEATURES = ["opioid_rate_lag"]
TARGET_COL = "opioid_rate"


def evaluate_baseline_linear_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Fit the CP2 baseline on the train frame and score it on the test frame."""
    train_clean = train_df.dropna(subset=BASELINE_FEATURES + [TARGET_COL]).copy()
    test_clean = test_df.dropna(subset=BASELINE_FEATURES + [TARGET_COL]).copy()

    scaler = StandardScaler()
    model = LinearRegression()
    model.fit(scaler.fit_transform(train_clean[BASELINE_FEATURES]), train_clean[TARGET_COL])
    predictions = model.predict(scaler.transform(test_clean[BASELINE_FEATURES]))

    return {
        "model": model,
        "scaler": scaler,
        "features": BASELINE_FEATURES,
        "r2": r2_score(test_clean[TARGET_COL], predictions),
        "rmse": np.sqrt(mean_squared_error(test_clean[TARGET_COL], predictions)),
        "predictions": pd.DataFrame(
            {"actual": test_clean[TARGET_COL].values, "predicted": predictions},
            index=test_clean.index,
        ),
    }

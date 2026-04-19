"""
workflow/modeling.py
--------------------
Stages 5 and 6 of the CheckPoint3 flow: train and evaluate the final Random Forest.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from .features import get_feature_columns


RF_PARAMS = {
    "n_estimators": 500,
    "random_state": 42,
    "n_jobs": -1,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
}

TARGET_COL = "opioid_rate"


def _validate_model_frame(df: pd.DataFrame) -> list[str]:
    features = get_feature_columns()
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing engineered features: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataframe must include target column '{TARGET_COL}'.")
    return features


def fit_random_forest(df: pd.DataFrame) -> RandomForestRegressor:
    """Fit the final model on a fully prepared training dataframe."""
    features = _validate_model_frame(df)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(df[features].fillna(0), df[TARGET_COL])
    return model


def evaluate_model(model, df: pd.DataFrame) -> dict:
    """Evaluate a trained model on a prepared holdout dataframe."""
    features = _validate_model_frame(df)
    X = df[features].fillna(0)
    y = df[TARGET_COL]
    y_pred = model.predict(X)
    return {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "predictions": pd.DataFrame({"actual": y.values, "predicted": y_pred}, index=df.index),
        "feature_importances": pd.Series(
            model.feature_importances_, index=features
        ).sort_values(ascending=False),
        "X_test": X,
        "y_test": y,
    }


def train_random_forest(df: pd.DataFrame, test_size: float = 0.2):
    """Train the final model on the engineered county feature set."""
    features = _validate_model_frame(df)
    X = df[features].fillna(0)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )

    return model, {
        "r2": r2,
        "rmse": rmse,
        "cv_scores": cv_scores,
        "predictions": pd.DataFrame({"actual": y_test.values, "predicted": y_pred}),
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test,
    }


def save_model(model, path: str = "outputs/models/rf_opioid_pricer.pkl") -> None:
    """Save a trained model to disk."""
    joblib.dump(model, path)


def load_model(path: str = "outputs/models/rf_opioid_pricer.pkl"):
    """Load a persisted model from disk."""
    return joblib.load(path)


def predict_counties(model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate county-level predictions and preserve geographic identifiers."""
    features = get_feature_columns()
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Prediction dataframe is missing engineered features: {missing}")

    id_candidates = ["county_fips", "county_name", "state"]
    available_ids = [column for column in id_candidates if column in df.columns]
    result = df[available_ids].copy() if available_ids else pd.DataFrame(index=df.index)
    result["predicted_rate"] = model.predict(df[features].fillna(0))
    result["actual_rate"] = df[TARGET_COL].values if TARGET_COL in df.columns else None
    return result.sort_values("predicted_rate", ascending=False)

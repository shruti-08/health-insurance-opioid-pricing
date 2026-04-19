"""
workflow/orchestration.py
-------------------------
High-level helpers that run the repository in the same order as CheckPoint3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .assemble import build_model_datasets
from .baseline import evaluate_baseline_linear_regression
from .features import engineer_features, get_feature_columns, validate_features
from .ingest import load_processed_cms, prepare_processed_cms
from .modeling import evaluate_model, fit_random_forest, predict_counties, save_model, train_random_forest
from .roi import calculate_roi
from .segmentation import assign_risk_tiers, summarize_tiers


def _ensure_path(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_prepared_frame_pipeline(df: pd.DataFrame, *, test_size: float = 0.2) -> dict:
    """Run the project on a single prepared dataframe with lag columns."""
    validate_features(df)
    engineered_df = engineer_features(df)
    model, final_model = train_random_forest(engineered_df, test_size=test_size)
    county_predictions = predict_counties(model, engineered_df)
    tiered_df = assign_risk_tiers(county_predictions)
    return {
        "mode": "prepared_frame",
        "input_rows": len(df),
        "feature_columns": get_feature_columns(),
        "model": model,
        "baseline": None,
        "final_model": final_model,
        "tiered_df": tiered_df,
        "tier_summary": summarize_tiers(tiered_df),
        "roi": calculate_roi(final_rmse=final_model["rmse"]),
    }


def run_notebook_style_pipeline(
    train_cms_path: str,
    test_cms_path: str,
    *,
    outlier_sigma: float = 3.0,
) -> dict:
    """Run the CP3 notebook flow using two CMS files from consecutive years."""
    processed_train_path = prepare_processed_cms(train_cms_path, year="train")
    processed_test_path = prepare_processed_cms(test_cms_path, year="test")
    train_processed = load_processed_cms(processed_train_path)
    test_processed = load_processed_cms(processed_test_path)

    df_train, df_test, cutoff = build_model_datasets(
        train_processed, test_processed, outlier_sigma=outlier_sigma
    )
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    baseline = evaluate_baseline_linear_regression(df_train, df_test)
    model = fit_random_forest(df_train)
    final_model = evaluate_model(model, df_test)
    county_predictions = predict_counties(model, df_test).merge(
        df_test[
            [
                column
                for column in [
                    "county_fips",
                    "opioid_rate_lag",
                    "opioid_share",
                    "la_rate",
                    "rate_1y_chg",
                ]
                if column in df_test.columns
            ]
        ],
        on="county_fips",
        how="left",
    )
    tiered_df = assign_risk_tiers(county_predictions)

    return {
        "mode": "notebook_style",
        "input_rows": len(df_test),
        "feature_columns": get_feature_columns(),
        "outlier_cutoff": cutoff,
        "processed_train_path": str(processed_train_path),
        "processed_test_path": str(processed_test_path),
        "model": model,
        "baseline": baseline,
        "final_model": final_model,
        "tiered_df": tiered_df,
        "tier_summary": summarize_tiers(tiered_df),
        "roi": calculate_roi(
            baseline_rmse=baseline["rmse"],
            final_rmse=final_model["rmse"],
        ),
    }


def save_pipeline_outputs(results: dict, *, predictions_out: str, model_out: str) -> tuple[Path, Path]:
    """Persist the tiered predictions CSV and trained model."""
    predictions_path = _ensure_path(predictions_out)
    model_path = _ensure_path(model_out)
    results["tiered_df"].to_csv(predictions_path, index=False)
    save_model(results["model"], str(model_path))
    return predictions_path, model_path

"""Workflow-stage modules that mirror the CheckPoint3 notebook flow."""

from .assemble import build_model_datasets
from .baseline import evaluate_baseline_linear_regression
from .features import engineer_features, get_feature_columns, validate_features
from .ingest import (
    CMS_NUMERIC_COLUMNS,
    CMS_RENAME_MAP,
    load_and_clean_cms,
    load_processed_cms,
    prepare_processed_cms,
    save_processed_cms,
)
from .modeling import (
    RF_PARAMS,
    TARGET_COL,
    evaluate_model,
    fit_random_forest,
    load_model,
    predict_counties,
    save_model,
    train_random_forest,
)
from .orchestration import run_notebook_style_pipeline, run_prepared_frame_pipeline
from .roi import ROIAssumptions, calculate_exposure, calculate_roi, sensitivity_analysis
from .segmentation import (
    PREMIUM_MULTIPLIERS,
    TIER_LABELS,
    assign_risk_tiers,
    find_optimal_k,
    get_top_high_risk,
    summarize_tiers,
)

__all__ = [
    "CMS_NUMERIC_COLUMNS",
    "CMS_RENAME_MAP",
    "PREMIUM_MULTIPLIERS",
    "RF_PARAMS",
    "ROIAssumptions",
    "TARGET_COL",
    "TIER_LABELS",
    "assign_risk_tiers",
    "build_model_datasets",
    "calculate_exposure",
    "calculate_roi",
    "engineer_features",
    "evaluate_baseline_linear_regression",
    "evaluate_model",
    "find_optimal_k",
    "fit_random_forest",
    "get_feature_columns",
    "get_top_high_risk",
    "load_and_clean_cms",
    "load_processed_cms",
    "load_model",
    "predict_counties",
    "prepare_processed_cms",
    "run_notebook_style_pipeline",
    "run_prepared_frame_pipeline",
    "save_processed_cms",
    "save_model",
    "sensitivity_analysis",
    "summarize_tiers",
    "train_random_forest",
    "validate_features",
]

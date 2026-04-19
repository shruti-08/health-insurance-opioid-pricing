"""Backward-compatible import wrapper for workflow model utilities."""

try:
    from .workflow.modeling import (
        RF_PARAMS,
        TARGET_COL,
        evaluate_model,
        fit_random_forest,
        load_model,
        predict_counties,
        save_model,
        train_random_forest,
    )
except ImportError:
    from workflow.modeling import (
        RF_PARAMS,
        TARGET_COL,
        evaluate_model,
        fit_random_forest,
        load_model,
        predict_counties,
        save_model,
        train_random_forest,
    )

__all__ = [
    "RF_PARAMS",
    "TARGET_COL",
    "evaluate_model",
    "fit_random_forest",
    "load_model",
    "predict_counties",
    "save_model",
    "train_random_forest",
]

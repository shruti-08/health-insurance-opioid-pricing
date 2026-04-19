"""Backward-compatible wrapper for the CP2 baseline model stage."""

try:
    from .workflow.baseline import BASELINE_FEATURES, TARGET_COL, evaluate_baseline_linear_regression
except ImportError:
    from workflow.baseline import BASELINE_FEATURES, TARGET_COL, evaluate_baseline_linear_regression

__all__ = ["BASELINE_FEATURES", "TARGET_COL", "evaluate_baseline_linear_regression"]

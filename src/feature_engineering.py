"""Backward-compatible import wrapper for workflow feature engineering."""

try:
    from .workflow.features import engineer_features, get_feature_columns, validate_features
except ImportError:
    from workflow.features import engineer_features, get_feature_columns, validate_features

__all__ = ["engineer_features", "get_feature_columns", "validate_features"]

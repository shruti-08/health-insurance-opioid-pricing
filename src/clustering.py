"""Backward-compatible import wrapper for workflow segmentation utilities."""

try:
    from .workflow.segmentation import (
        PREMIUM_MULTIPLIERS,
        TIER_LABELS,
        assign_risk_tiers,
        find_optimal_k,
        get_top_high_risk,
        summarize_tiers,
    )
except ImportError:
    from workflow.segmentation import (
        PREMIUM_MULTIPLIERS,
        TIER_LABELS,
        assign_risk_tiers,
        find_optimal_k,
        get_top_high_risk,
        summarize_tiers,
    )

__all__ = [
    "PREMIUM_MULTIPLIERS",
    "TIER_LABELS",
    "assign_risk_tiers",
    "find_optimal_k",
    "get_top_high_risk",
    "summarize_tiers",
]

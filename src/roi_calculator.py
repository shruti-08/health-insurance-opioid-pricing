"""Backward-compatible import wrapper for workflow ROI calculations."""

try:
    from .workflow.roi import (
        ROIAssumptions,
        calculate_exposure,
        calculate_roi,
        sensitivity_analysis,
    )
except ImportError:
    from workflow.roi import (
        ROIAssumptions,
        calculate_exposure,
        calculate_roi,
        sensitivity_analysis,
    )

__all__ = [
    "ROIAssumptions",
    "calculate_exposure",
    "calculate_roi",
    "sensitivity_analysis",
]

"""Backward-compatible wrapper for stage 2 dataset assembly helpers."""

try:
    from .workflow.assemble import build_model_datasets
except ImportError:
    from workflow.assemble import build_model_datasets

__all__ = ["build_model_datasets"]

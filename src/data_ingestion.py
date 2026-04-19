"""Backward-compatible wrapper for stage 1 data ingestion helpers."""

try:
    from .workflow.ingest import (
        CMS_NUMERIC_COLUMNS,
        CMS_RENAME_MAP,
        load_and_clean_cms,
        load_processed_cms,
        prepare_processed_cms,
        save_processed_cms,
    )
except ImportError:
    from workflow.ingest import (
        CMS_NUMERIC_COLUMNS,
        CMS_RENAME_MAP,
        load_and_clean_cms,
        load_processed_cms,
        prepare_processed_cms,
        save_processed_cms,
    )

__all__ = [
    "CMS_NUMERIC_COLUMNS",
    "CMS_RENAME_MAP",
    "load_and_clean_cms",
    "load_processed_cms",
    "prepare_processed_cms",
    "save_processed_cms",
]

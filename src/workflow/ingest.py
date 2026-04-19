"""
workflow/ingest.py
------------------
Stage 1 of the CheckPoint3 flow: load and clean raw CMS opioid geography data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CMS_RENAME_MAP = {
    "prscrbr_geo_cd": "county_fips",
    "prscrbr_geo_desc": "county_name",
    "tot_clms": "total_claims",
    "tot_opioid_clms": "opioid_claims",
    "tot_prscrbrs": "prescribers",
    "tot_opioid_prscrbrs": "opioid_prescribers",
    "opioid_prscrbng_rate": "opioid_rate",
    "opioid_prscrbng_rate_1y_chg": "rate_1y_chg",
    "opioid_prscrbng_rate_5y_chg": "rate_5y_chg",
    "la_opioid_prscrbng_rate": "la_rate",
    "la_opioid_prscrbng_rate_1y_chg": "la_rate_1y_chg",
}

CMS_NUMERIC_COLUMNS = [
    "opioid_rate",
    "rate_1y_chg",
    "rate_5y_chg",
    "la_rate",
    "la_rate_1y_chg",
    "total_claims",
    "opioid_claims",
    "prescribers",
    "opioid_prescribers",
]


def load_and_clean_cms(filepath: str | Path, year: int | str | None = None) -> pd.DataFrame:
    """Load a CMS geography CSV and normalize it for downstream modeling stages."""
    df = pd.read_csv(filepath)

    if "Prscrbr_Geo_Lvl" in df.columns:
        df = df[df["Prscrbr_Geo_Lvl"] == "County"].copy()
    if "Breakout" in df.columns:
        df = df[df["Breakout"] == "Overall"].copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns={k: v for k, v in CMS_RENAME_MAP.items() if k in df.columns})

    if "county_fips" not in df.columns:
        raise ValueError("CMS file is missing the county FIPS column after standardization.")

    df["county_fips"] = pd.to_numeric(df["county_fips"], errors="coerce")
    df = df.dropna(subset=["county_fips"])
    df["county_fips"] = df["county_fips"].astype(int).astype(str).str.zfill(5)
    df = df[~df["county_fips"].str.startswith("72")].copy()

    if "opioid_rate" in df.columns:
        df["opioid_rate"] = pd.to_numeric(df["opioid_rate"], errors="coerce")
        df = df.dropna(subset=["opioid_rate"])

    for column in CMS_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    if year is not None:
        print(f"Loaded CMS {year}: {len(df):,} counties")

    return df


def save_processed_cms(df: pd.DataFrame, filepath: str | Path) -> Path:
    """Save a cleaned CMS dataframe into the processed data folder."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def prepare_processed_cms(
    raw_path: str | Path,
    processed_path: str | Path | None = None,
    *,
    year: int | str | None = None,
) -> Path:
    """
    Clean a raw CMS file and write the cleaned result into data/processed.

    When processed_path is omitted, the output file is derived from the raw filename.
    Example: data/raw/cms_2020.csv -> data/processed/cms_2020_cleaned.csv
    """
    raw_path = Path(raw_path)
    if processed_path is None:
        processed_path = Path("data/processed") / f"{raw_path.stem}_cleaned.csv"
    cleaned_df = load_and_clean_cms(raw_path, year=year)
    output_path = save_processed_cms(cleaned_df, processed_path)
    if year is not None:
        print(f"Saved processed CMS {year}: {output_path}")
    return output_path


def load_processed_cms(filepath: str | Path) -> pd.DataFrame:
    """Load a cleaned CMS file from data/processed and preserve county FIPS as strings."""
    return pd.read_csv(filepath, dtype={"county_fips": str})

"""
run_pipeline.py
---------------
Command-line entry point for the end-to-end pricing pipeline.

Usage
-----
python -m src.run_pipeline --input path/to/county_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .workflow.orchestration import (
    run_notebook_style_pipeline,
    run_prepared_frame_pipeline,
    save_pipeline_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the opioid risk pricing pipeline on a county-level CSV."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Path to a prepared county-level CSV that already includes lag columns.",
    )
    input_group.add_argument(
        "--train-cms",
        help="Notebook-style mode: CMS file for the training year, e.g. 2020.",
    )
    parser.add_argument(
        "--test-cms",
        help="Notebook-style mode: CMS file for the scoring year, e.g. 2021.",
    )
    parser.add_argument(
        "--predictions-out",
        default="outputs/predictions/county_risk_predictions.csv",
        help="Where to save the county-level tiered predictions CSV.",
    )
    parser.add_argument(
        "--model-out",
        default="outputs/models/rf_opioid_pricer.pkl",
        help="Where to save the trained Random Forest model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Prepared-frame mode only: fraction of rows reserved for the test split.",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=3.0,
        help="Notebook-style mode only: number of standard deviations used for outlier trimming.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.train_cms and not args.test_cms:
        raise ValueError("--test-cms is required when --train-cms is provided.")
    if args.test_cms and not args.train_cms:
        raise ValueError("--train-cms is required when --test-cms is provided.")

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        import pandas as pd

        df = pd.read_csv(input_path)
        results = run_prepared_frame_pipeline(df, test_size=args.test_size)
    else:
        if not Path(args.train_cms).exists():
            raise FileNotFoundError(f"Training CMS CSV not found: {args.train_cms}")
        if not Path(args.test_cms).exists():
            raise FileNotFoundError(f"Testing CMS CSV not found: {args.test_cms}")
        results = run_notebook_style_pipeline(
            args.train_cms,
            args.test_cms,
            outlier_sigma=args.outlier_sigma,
        )

    predictions_out, model_out = save_pipeline_outputs(
        results,
        predictions_out=args.predictions_out,
        model_out=args.model_out,
    )

    print("\n=== Pipeline Complete ===")
    print(f"Pipeline mode           : {results['mode']}")
    print(f"Input rows              : {results['input_rows']}")
    print(f"Feature columns used    : {', '.join(results['feature_columns'])}")
    if "processed_train_path" in results and "processed_test_path" in results:
        print(f"Processed train data    : {results['processed_train_path']}")
        print(f"Processed test data     : {results['processed_test_path']}")
    if results["baseline"] is not None:
        print(f"Baseline RMSE           : {results['baseline']['rmse']:.3f}")
        print(f"Baseline R^2            : {results['baseline']['r2']:.3f}")
    print(f"Final Model R^2         : {results['final_model']['r2']:.3f}")
    print(f"Final Model RMSE        : {results['final_model']['rmse']:.3f}")
    print(f"Predictions saved to    : {predictions_out}")
    print(f"Model saved to          : {model_out}")
    print(f"Estimated annual ROI    : ${results['roi']['annual_savings_M']:.1f}M")

    print("\n=== Tier Summary ===")
    print(results["tier_summary"].reset_index().to_string(index=False))


if __name__ == "__main__":
    main()

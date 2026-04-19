"""
workflow/roi.py
---------------
Stage 9 of the CheckPoint3 flow: translate model improvement into business ROI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ROIAssumptions:
    avg_claim_cost: float = 4_500
    claims_per_patient_yr: int = 12
    patients_per_county: int = 800
    counties: int = 3_031


def calculate_exposure(rmse: float, assumptions: ROIAssumptions | None = None) -> float:
    """Calculate annual mispricing exposure in dollars from RMSE."""
    assumptions = assumptions or ROIAssumptions()
    return (
        (rmse / 100)
        * assumptions.patients_per_county
        * assumptions.claims_per_patient_yr
        * assumptions.avg_claim_cost
        * assumptions.counties
    )


def calculate_roi(
    baseline_rmse: float = 0.724,
    final_rmse: float = 0.432,
    assumptions: ROIAssumptions | None = None,
) -> dict:
    """Compare baseline and final model exposure to estimate annual savings."""
    assumptions = assumptions or ROIAssumptions()
    baseline_exposure = calculate_exposure(baseline_rmse, assumptions)
    final_exposure = calculate_exposure(final_rmse, assumptions)
    savings = baseline_exposure - final_exposure
    return {
        "baseline_rmse": baseline_rmse,
        "final_rmse": final_rmse,
        "rmse_reduction_pct": ((baseline_rmse - final_rmse) / baseline_rmse) * 100,
        "baseline_exposure_M": baseline_exposure / 1_000_000,
        "final_exposure_M": final_exposure / 1_000_000,
        "annual_savings_M": savings / 1_000_000,
        "exposure_reduction_pct": (savings / baseline_exposure) * 100,
        "assumptions": assumptions,
    }


def sensitivity_analysis(assumptions: ROIAssumptions | None = None) -> list[dict]:
    """Return ROI outcomes under a few alternate assumption sets."""
    assumptions = assumptions or ROIAssumptions()
    scenarios = [
        ("Conservative", 600, 10),
        ("Base case", 800, 12),
        ("Optimistic", 1_000, 15),
    ]
    results = []
    for label, patients, claims in scenarios:
        scenario_assumptions = ROIAssumptions(
            avg_claim_cost=assumptions.avg_claim_cost,
            claims_per_patient_yr=claims,
            patients_per_county=patients,
            counties=assumptions.counties,
        )
        outcome = calculate_roi(assumptions=scenario_assumptions)
        outcome["scenario"] = label
        results.append(outcome)
    return results

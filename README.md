# health-insurance-opioid-pricing

**Predicting county-level opioid prescribing rates to enable actuarially sound insurance premium pricing.**

> *"Where you prescribed opioids last year — and how fast that is changing — predicts next year's rate more accurately than any demographic. Geography is destiny."*

---

## The Business Problem

Opioid claims are a massive cost center for health insurers. Traditional premium pricing relies on static demographic snapshots (age, poverty rate, unemployment) — but these explain **less than 5% of county-level opioid prescribing variance**.

This project builds a machine learning pipeline that:
1. **Predicts** next-year opioid prescribing rates for all 3,031 US counties
2. **Segments** counties into actionable Low / Medium / High risk tiers
3. **Prices** insurance premiums accordingly to reduce mispricing exposure

---

## Key Results

| Metric | CP2 Baseline (Linear Regression) | CP3 Final (Random Forest) |
|--------|----------------------------------|---------------------------|
| R² Score | 0.783 | **0.923** |
| RMSE | 0.724 | **0.432** |
| Variance Explained | 78.3% | **92.3%** |

- **40.4% reduction** in prediction error
- **$382M annual savings** from reduced premium mispricing across 3,031 counties
- **3 premium tiers** derived from K-Means clustering (k=3)

---

## Premium Tier Output

| Tier | Counties | Avg Predicted Rate | Premium Multiplier |
|------|----------|-------------------|-------------------|
| Low Risk | 974 | 2.57 | 0.85× |
| Medium Risk | 1,435 | 4.14 | 1.00× |
| High Risk | 622 | 6.34 | 1.25× |

---

## The Key Insight — Economic Momentum

Static Census demographics gave **0% marginal lift**. What actually predicts opioid risk:

- `opioid_lag_sq` — squared prior-year rate (captures persistence in high-risk counties)
- `opioid_share` — opioid claims / total claims (prescribing intensity)
- `la_share` — long-acting opioid share (addiction severity proxy)
- `prescriber_ratio` — opioid prescribers / total prescribers (provider behavior)
- `rate_1y_chg`, `rate_5y_chg` — year-over-year momentum (the core insight)

**Change velocity predicts future risk better than absolute levels.**

---

## Project Flow

The notebook you shared follows this sequence:

1. Load and clean CMS opioid geography data
2. Build train/test sets with lag fields
3. Engineer CP3 momentum features
4. Fit the CP2 baseline linear regression
5. Fit the CP3 final Random Forest
6. Evaluate and compare model performance
7. Segment counties into premium tiers with K-Means
8. Estimate annual ROI from RMSE reduction

The repository is now arranged around that same flow.

## Project Structure

```
health-insurance-opioid-pricing/
├── data/
│   ├── raw/            ← original CMS & ACS downloads (not committed)
│   ├── processed/      ← cleaned, merged datasets
│   └── external/       ← ACS Census demographic files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_linear_regression.ipynb
│   ├── 03_random_forest.ipynb
│   └── 04_kmeans_clustering.ipynb
├── src/
│   ├── workflow/
│   │   ├── ingest.py
│   │   ├── assemble.py
│   │   ├── features.py
│   │   ├── baseline.py
│   │   ├── modeling.py
│   │   ├── segmentation.py
│   │   ├── roi.py
│   │   └── orchestration.py
│   ├── run_pipeline.py
│   └── thin compatibility wrappers
├── outputs/
│   ├── figures/        ← charts and maps
│   ├── models/         ← saved Random Forest model (.pkl)
│   └── predictions/    ← county risk scores CSV
├── docs/
│   ├── final_report.pdf
│   └── CP3_slides.pptx
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/health-insurance-opioid-pricing.git
cd health-insurance-opioid-pricing
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the data
- **CMS Medicare Part D Prescribers** — [data.cms.gov](https://data.cms.gov)
- **ACS 5-Year Estimates** — [census.gov/programs-surveys/acs](https://www.census.gov/programs-surveys/acs)

Place raw CMS files in `data/raw/`. The pipeline will clean them, drop invalid rows, write cleaned versions into `data/processed/`, and then use those processed files for downstream modeling.

### 4. Run the notebooks in order
```
01_eda.ipynb               ← Exploratory Data Analysis (CP1)
02_linear_regression.ipynb ← Baseline model (CP2)
03_random_forest.ipynb     ← Final model (CP3)
04_kmeans_clustering.ipynb ← Premium tier segmentation
```

### 5. Build the notebook flow directly in code
```python
from src.workflow.ingest import load_processed_cms, prepare_processed_cms
from src.workflow.assemble import build_model_datasets
from src.workflow.features import engineer_features
from src.workflow.baseline import evaluate_baseline_linear_regression
from src.workflow.modeling import fit_random_forest, evaluate_model, predict_counties
from src.workflow.segmentation import assign_risk_tiers
from src.workflow.roi import calculate_roi

processed_2020 = prepare_processed_cms("data/raw/cms_2020.csv", year=2020)
processed_2021 = prepare_processed_cms("data/raw/cms_2021.csv", year=2021)

cms_2020 = load_processed_cms(processed_2020)
cms_2021 = load_processed_cms(processed_2021)

df_train, df_test, cutoff = build_model_datasets(cms_2020, cms_2021)
df_train = engineer_features(df_train)
df_test = engineer_features(df_test)

baseline = evaluate_baseline_linear_regression(df_train, df_test)
rf = fit_random_forest(df_train)
final_model = evaluate_model(rf, df_test)
tiered_df = assign_risk_tiers(predict_counties(rf, df_test))
roi = calculate_roi(baseline["rmse"], final_model["rmse"])
```

### 6. Run the CLI in either mode
```bash
python -m src.run_pipeline --input data/processed/county_features.csv
python -m src.run_pipeline --train-cms data/raw/cms_2020.csv --test-cms data/raw/cms_2021.csv
```

---

## ROI Calculation

```
Annual Savings = (RMSE_baseline - RMSE_final) / 100
                 × patients_per_county (800)
                 × claims_per_patient_per_year (12)
                 × avg_claim_cost ($4,500)
                 × counties_scored (3,031)

= $382.6M annually
```

> **Note:** 800 patients/county and 12 claims/year are national Medicare averages used as conservative proxies. A production deployment would substitute actual book-of-business data per county.

---

## Team

**Team 6 — Actuarial Risk Squad**
Aashish Wagle · Lavannya Patil · Shruti Deulgaonkar

ITCS 6100 · Spring 2026 · UNC Charlotte

---

## License

For academic use only. Data sourced from CMS and US Census Bureau.

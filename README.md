# health-insurance-opioid-pricing

**Predicting county-level opioid prescribing rates to enable actuarially sound insurance premium pricing.**

> *"Where you prescribed opioids last year — and how fast that is changing — predicts next year's rate more accurately than any demographic."*

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

## Project Structure

```
rx-risk-pricer/
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
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── clustering.py
│   └── roi_calculator.py
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
git clone https://github.com/YOUR_USERNAME/rx-risk-pricer.git
cd rx-risk-pricer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the data
- **CMS Medicare Part D Prescribers** — [data.cms.gov](https://data.cms.gov)
- **ACS 5-Year Estimates** — [census.gov/programs-surveys/acs](https://www.census.gov/programs-surveys/acs)

Place raw files in `data/raw/`.

### 4. Run the notebooks in order
```
01_eda.ipynb               ← Exploratory Data Analysis (CP1)
02_linear_regression.ipynb ← Baseline model (CP2)
03_random_forest.ipynb     ← Final model (CP3)
04_kmeans_clustering.ipynb ← Premium tier segmentation
```

### 5. Or run the full pipeline via src/
```python
from src.feature_engineering import engineer_features
from src.model_training import train_random_forest
from src.clustering import assign_risk_tiers
from src.roi_calculator import calculate_roi

df = engineer_features(raw_df)
model, predictions = train_random_forest(df)
tiered_df = assign_risk_tiers(predictions)
roi = calculate_roi(baseline_rmse=0.724, final_rmse=0.432)
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
Shruti Deulgaonkar · Aashish Wagle · Lavannya Patil

ITCS 6100 · Spring 2026 · UNC Charlotte

---

## License

For academic use only. Data sourced from CMS and US Census Bureau.

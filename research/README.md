# research/

Exploratory data analysis, feature engineering, and baseline modeling on the Yakimov et al. (2024) fingernail dataset.

## Notebooks

| Notebook | Description | Status |
|----------|-------------|--------|
| `01_eda.ipynb` | Dataset loading, class distribution, image quality assessment | ✅ Done |
| `02_feature_extraction.ipynb` | ROI extraction, color-space analysis (RGB, LAB, HSV), 51 features | ✅ Done |
| `03_baseline_models.ipynb` | Traditional ML baselines (Ridge, SVR, Random Forest, Gradient Boosting) | ✅ Done |

## Baseline Results (Notebook 03)

| Model | Test MAE (g/dL) | Test R² |
|-------|-----------------|--------|
| Ridge Regression | 1.575 | 0.384 |
| Random Forest | 1.610 | 0.307 |
| Gradient Boosting | 1.649 | 0.288 |
| Lasso Regression | 1.692 | 0.324 |
| SVR (RBF) | 1.774 | 0.231 |

**Top predictive features** (Gradient Boosting importance): `skin_lab_B_lab_std`, `skin_lab_A_mean`, `nail_lab_L_mean`, `nail_rgb_R_mean`

## Key Findings

- **Nail-bed green channel** and **LAB luminance** are most correlated with Hb (|r| ≈ 0.52)
- **LAB A-channel** (redness) shows the strongest positive correlation — redder nails → higher Hb
- Contrast features (nail vs. skin) provide moderate signal
- Hb range in dataset: 4.4–16.9 g/dL (includes severe anemia cases)

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

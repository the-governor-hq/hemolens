# research/

Exploratory data analysis, metadata profiling, and baseline modeling on the Yakimov et al. (2024) fingernail dataset.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Dataset loading, class distribution, image quality assessment |
| `02_feature_extraction.ipynb` | ROI extraction, color-space analysis (LAB, HSV), nail-bed segmentation |
| `03_baseline_models.ipynb` | Traditional ML baselines (Ridge, SVR, XGBoost) on hand-crafted features |

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

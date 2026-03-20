# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS only (for XGBoost):
brew install libomp

# Train all models
PYTHONPATH=src python src/train.py

# Run inference
PYTHONPATH=src python src/predict.py

# Evaluate models
PYTHONPATH=src python src/evaluate.py

# Run tests
python tests/test_preprocessing.py
```

## Architecture

This system predicts next-day PM2.5 (µg/m³) for a specific monitoring station (Station 10T, Bangkok) using regression models trained on time-series features from Thai air quality data.

**Data flow:**
```
Raw Excel (data/raw/) → data_loader.py → preprocessing.py → feature_engineering.py → train.py → models/
                                                                                              ↓
                                                                                        predict.py (inference)
```

**Pipeline modules (`src/`):**
- `data_loader.py` — Loads per-station data from Excel files with Thai-language metadata sheets
- `preprocessing.py` — ffill/bfill for missing values, removes values outside [0, 500] µg/m³
- `feature_engineering.py` — Builds 17 features: lag features (`pm25_lag_1/2/3/5/7`), rolling mean/std over 3/7/14 days, time features (day_of_week, month, day_of_year, is_weekend), and change features (diff, pct_change). All use `shift(1)` to prevent data leakage.
- `train.py` — Orchestrates full training: loads 2024 data (train) and 2025 data (test), runs GridSearchCV with `TimeSeriesSplit(n_splits=3)` for Ridge, Random Forest, and XGBoost. Saves `.joblib` models and `feature_columns.json`.
- `evaluate.py` — Computes MAE (primary), RMSE, R²; results written to `results/experiment_results.csv`
- `predict.py` — Standalone inference using saved model and `feature_columns.json`

**Configuration:** All paths, station ID, feature engineering parameters, hyperparameter grids, and random state (42) are centralized in `configs/config.yaml`.

**Selected model:** Random Forest (MAE: 4.57 µg/m³, R²: 0.783) outperforms baseline Linear Regression (MAE: 5.13) on 2025 test data.

**Key constraints:**
- Always use `PYTHONPATH=src` when running scripts directly (modules import each other without a package install)
- Feature engineering drops NaN rows from lag/rolling operations — training set shrinks from 366 → 359 days, test from 181 → 174 days
- Temporal data split: validation starts 2024-11-01; never shuffle time-series data

# PM2.5 Prediction ML System

A machine learning system for predicting PM2.5 (fine particulate matter) concentration levels using historical air quality monitoring data from Thailand's Pollution Control Department (กรมควบคุมมลพิษ).

## Project Overview

PM2.5 air pollution is a critical public health concern in Thailand, particularly during the dry season (November–April). This project builds a regression-based ML system to predict daily PM2.5 levels at monitoring stations, using historical patterns and time-series features.

**Objective:** Predict next-day PM2.5 concentration (µg/m³) for a given station using lag features, rolling statistics, and calendar features.

## Dataset

| Item | Detail |
|------|--------|
| Source | กรมควบคุมมลพิษ (Pollution Control Department, Thailand) |
| Files | `PM2.5(2024).xlsx` (training), `PM2.5(2025).xlsx` (testing) |
| Stations | 96 monitoring stations across Thailand |
| Granularity | Daily PM2.5 values per station |
| Selected Station | **10T** — เคหะชุมชนคลองจั่น, เขตบางกะปิ, กทม. |

## Repository Structure

```
pm25-prediction-ml-system/
├── README.md                         # This file
├── .gitignore
├── requirements.txt                  # Python dependencies
├── configs/
│   └── config.yaml                   # Centralized configuration
├── data/
│   ├── raw/                          # Original Excel files
│   │   ├── PM2.5(2024).xlsx          # Training data (year 2024)
│   │   └── PM2.5(2025).xlsx          # Test data (year 2025)
│   └── processed/                    # Cleaned CSV files
├── notebooks/
│   └── 01_eda.ipynb                  # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Data loading utilities
│   ├── preprocessing.py              # Missing value & outlier handling
│   ├── feature_engineering.py        # Feature creation pipeline
│   ├── train.py                      # Model training pipeline
│   ├── evaluate.py                   # Evaluation metrics
│   └── predict.py                    # Inference pipeline (separated)
├── models/                           # Saved trained models (.joblib)
├── reports/
│   └── progress_report_1.md          # Progress Report 1
├── results/
│   └── experiment_results.csv        # Model comparison metrics
└── tests/
    └── test_preprocessing.py         # Unit tests
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd pm25-prediction-ml-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# macOS only: XGBoost requires OpenMP
brew install libomp
```

## Usage

### 1. Train All Models

```bash
PYTHONPATH=src python src/train.py
```

This will:
- Load and preprocess training data (2024) and test data (2025)
- Engineer 17 time-series features
- Train 4 models: Linear Regression, Ridge, Random Forest, XGBoost
- Perform hyperparameter tuning with TimeSeriesSplit CV
- Save models to `models/` and results to `results/`

### 2. View Results

```bash
PYTHONPATH=src python src/evaluate.py
```

### 3. Run Inference

```bash
PYTHONPATH=src python src/predict.py
```

### 4. Run Tests

```bash
python tests/test_preprocessing.py
```

## Experiment Results

Station: **10T** (เคหะชุมชนคลองจั่น, บางกะปิ, กทม.)
Train: 2024 data (366 days) → Test: 2025 data (174 days after feature engineering)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression (Baseline) | 5.1348 | 6.7493 | 0.7726 |
| Ridge Regression | 4.8286 | **6.5294** | **0.7871** |
| Random Forest | **4.5702** | 6.5961 | 0.7828 |
| XGBoost | 4.9735 | 7.3464 | 0.7305 |

**Best Model:** Random Forest (lowest MAE = 4.57 µg/m³, best primary metric)

## Features (17 total)

| Category | Features |
|----------|----------|
| Lag features | `pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3`, `pm25_lag_5`, `pm25_lag_7` |
| Rolling statistics | `pm25_rolling_mean_3/7/14`, `pm25_rolling_std_3/7/14` |
| Time features | `day_of_week`, `month`, `day_of_year`, `is_weekend` |
| Change features | `pm25_diff_1`, `pm25_pct_change_1` |

## Configuration

All parameters are centralized in `configs/config.yaml`:
- Station selection
- Feature engineering settings (lag days, rolling windows)
- Model hyperparameter grids
- Data split dates
- Output paths

## Model Versioning

- Trained models saved as `.joblib` files in `models/`
- Feature columns saved as `models/feature_columns.json` for reproducible inference
- Configuration tracked in `configs/config.yaml`

## Tech Stack

- **Python 3.14**
- **pandas** / **numpy** — data manipulation
- **scikit-learn** — ML models, evaluation, tuning
- **XGBoost** — gradient boosting
- **matplotlib** / **seaborn** — visualization
- **PyYAML** — configuration management
- **joblib** — model serialization

## License

This project is for academic purposes (ML Systems course).
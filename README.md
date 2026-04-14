# Turbofan RUL Prediction using CMAPSS

This project predicts **Remaining Useful Life (RUL)** for turbofan engines using the NASA CMAPSS dataset. It includes an end-to-end workflow covering exploratory data analysis, condition-aware preprocessing, feature engineering, forecasting with classical machine learning models, sequence modeling with LSTM, and a local Flask-based web application for inference.

The goal is to compare traditional regression approaches with sequence-based deep learning for predictive maintenance.

---

## Project Overview

The objective is to estimate **how many operational cycles remain before engine failure** using:

- operational settings
- multivariate sensor readings

The project includes:

- exploratory analysis of engine degradation behavior
- operating-condition identification using **KMeans**
- condition-wise normalization of sensors
- feature engineering and combined dataset creation
- model training and evaluation
- grouped backtesting with **GroupKFold** to avoid leakage
- final evaluation on official CMAPSS test sets
- saving trained models and preprocessing artifacts
- serving predictions through a Flask-based local web app

---

## Models Implemented

- **Linear Regression**
- **Random Forest Regressor**
- **LSTM Regressor (PyTorch)**

---

## Modeling Design

| Feature | Value |
|---|---|
| Dataset | NASA CMAPSS (`FD001` to `FD004`) |
| Target | Remaining Useful Life (RUL) |
| Sequence Length | `30` |
| Final Feature Count | `22` |
| LSTM Input Shape | `(samples, 30, 22)` |
| LR/RF Input Shape | `(samples, 660)` |
| Validation Strategy | `GroupKFold` by `unit_id` |
| Test Strategy | last window per engine |

---

## Results Summary

### Backtesting Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 34.88 | 47.87 | 0.6190 |
| Random Forest | 29.97 | 44.43 | 0.6718 |
| LSTM | **29.21** | **43.13** | **0.6907** |

### Official Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 27.25 | 34.19 | 0.5518 |
| Random Forest | **22.99** | **31.83** | **0.6115** |
| LSTM | 24.50 | 34.03 | 0.5562 |

### Observations

- **LSTM performed best during grouped backtesting**
- **Random Forest generalized best on the official combined test set**
- **LSTM captured temporal patterns effectively through sequence modeling**
- **Grouped validation was important to prevent engine-level leakage from sliding windows**

---

## Repository Structure

```text
CMAPSS_Foreasting/
│
├── backend/
│   ├── app.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── state.py
│   ├── inference.py
│   └── templates/
│       └── index.html
│
├── notebooks/
│   ├── Turbofan.ipynb
│   └── Turbofan_forecasting.ipynb
│
├── CMAPSSData/
│   ├── train_*.txt
│   ├── test_*.txt
│   └── RUL_*.txt
│
├── combined_data/
│   ├── train_combined.csv
│   ├── test_combined.csv
│   ├── test_rul_combined.csv
│   ├── combined_features.json
│   └── preprocessing_summary.json
│
├── saved_models/
│   ├── final_linear_regression.joblib
│   ├── final_lstm_optimized.pth
│   ├── final_lstm_optimized_metadata.joblib
│   ├── final_lstm_optimized_metadata.json
│   ├── final_random_forest.joblib
│   ├── model_metadata.joblib
│   ├── model_metadata.json
│   └── preprocessing_artifacts.joblib
│
├── backtest_outputs/
├── final_test_outputs/
│
├── requirements.txt
├── README.md
└── .gitignore

# Turbofan RUL Prediction using CMAPSS

This project predicts **Remaining Useful Life (RUL)** for turbofan engines using the NASA **CMAPSS** dataset. It includes an end-to-end workflow covering exploratory data analysis, condition-aware preprocessing, feature engineering, forecasting with classical machine learning models, sequence modeling with LSTM, and a local Flask-based web application for inference.

The goal is to compare traditional regression approaches with sequence-based deep learning for predictive maintenance and deploy the final inference workflow through a usable local API and browser interface.

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
- API testing through **Postman**
- API documentation through **Swagger**

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

## New Deployment Features Added

The project was extended beyond training notebooks to include a full local prediction workflow.

### Web UI Features

The browser-based interface supports:

- single-row streaming prediction
- compare-all-models prediction
- rolling 30-cycle progress tracking
- prediction history table
- comparison chart for all three models
- file upload prediction from CSV
- reset engine state
- reset all in-memory states
- raw JSON response viewer
- direct access to Swagger API docs from the UI

### File Upload Prediction

Users can upload a CSV file containing sensor readings and request prediction using the last `N` rows.

The file upload workflow:

- accepts CSV input from the UI or API
- filters by selected `engine_id` if present
- validates required columns
- preprocesses each row using saved preprocessing artifacts
- builds a fixed 30-row model window
- pads or truncates rows when necessary
- returns a final RUL prediction

### Compare-All-Models Support

The backend also supports predicting with all three models at once using the same streaming update. This allows quick comparison of:

- Linear Regression
- Random Forest
- LSTM

---

## Local Web Application

A Flask backend is provided for local inference and testing.

### Backend Capabilities

- load saved ML and LSTM models once at startup
- preprocess raw streaming rows
- maintain rolling 30-cycle engine history in memory
- support prediction from:
  - one sensor row at a time
  - a full processed window
  - an uploaded CSV file
- return predictions in JSON format

### Main API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Browser UI |
| `/health` | GET | Health check |
| `/models` | GET | Model and preprocessing metadata |
| `/predict/window` | POST | Predict from full processed 30-row window |
| `/predict/stream` | POST | Predict from one streaming raw row |
| `/predict/stream/all` | POST | Predict with all three models on one shared stream update |
| `/predict/file` | POST | Predict from uploaded CSV file |
| `/engines/reset` | POST | Reset one engine state |
| `/engines/reset_all` | POST | Reset all engine states |
| `/apidocs/` | GET | Swagger API documentation |

---

## Swagger Documentation

Swagger documentation is integrated using **Flasgger**.

After starting the Flask app, open:

```text
http://127.0.0.1:5000/apidocs/

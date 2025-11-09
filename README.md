# Optiver---NASDAQ-Equities-Pricing
GRU + XGBoost for Optiver Trading Forecasting

This repository explores a series of machine learning and deep learning models for predicting price movement (target) in the Optiver Trading at the Close dataset. It includes experiments with gradient boosting models, a GRU-based neural network, and a hybrid GRU–XGBoost architecture.

📘 Overview

The notebook tests and compares several modeling strategies for time-series financial data:

Model	Description	Notes
LightGBM	Gradient boosting using leaf-wise trees	Fast, efficient baseline
XGBoost	Boosted decision trees	Robust, handles tabular features well
CatBoost	Categorical boosting	Minimal preprocessing
GRU	Gated Recurrent Unit network	Captures temporal patterns
GRU + XGBoost	Hybrid approach	GRU encodes sequence → XGBoost predicts target

🧩 Data Preparation

Load data from optiver-trading-at-the-close.zip

Create engineered features:

Bid-ask spread: ask_price - bid_price

Imbalance signed: imbalance_size * imbalance_buy_sell_flag

Lagged features: wap_lag1, target_lag1

Split data by time (no shuffle)

⚙️ Training and Evaluation

Metric: Mean Squared Error (MSE)

Validation: 80/20 time-based split

Comparison: Evaluate performance across models

🚀 Hybrid Model: GRU + XGBoost

The hybrid model combines:

GRU encoder: learns temporal embeddings of trading features.

XGBoost regressor: predicts target values from GRU-derived representations.

This approach leverages the sequential learning capacity of GRUs and the predictive strength of tree ensembles.

📈 Results (Example)
Best Result with GRU_XGBoost - RMSE ~ 7.8

🧠 Requirements
pip install numpy pandas scikit-learn lightgbm xgboost catboost torch

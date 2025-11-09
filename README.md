# Optiver---NASDAQ-Equities-Pricing

Model Comparison for Optiver Volatility Prediction
🧠 Overview

This project experiments with multiple gradient boosting methods for predicting financial metrics (e.g., realized volatility) using the Optiver dataset. It includes feature engineering, model training, and performance comparison.

🚀 Methods Tested

Three state-of-the-art gradient boosting models were benchmarked:

LightGBM, XGBoost, CatBoost, Hybrid GRU-XGBoost

Each model was trained on the same processed dataset, allowing direct comparison of performance metrics.

🧩 Feature Engineering

Features included:

Bid-ask spread: ask_price - bid_price

Signed order imbalance: imbalance_size * imbalance_buy_sell_flag

Lagged features: 1-step lag per stock to capture temporal effects

Aggregations: Mean, std, and percentile-based summaries by stock_id

⚙️ Model Training

Data split into train/test sets by stock_id or time-based splits.

Standardized training pipeline across models for fair comparison.

Metrics: RMSE (Root Mean Squared Error) and R².

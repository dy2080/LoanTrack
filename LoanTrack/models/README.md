# 💾 Trained Model Artifacts

This folder stores serialized machine learning models (`.pkl` files) generated during training.

- `xgboost_loan_model.pkl`: Best production model (XGBoost + scaler + label encoders)
- `with_log_xgboost_loan_model.pkl`: Model trained with log1p feature transformation
- `no_log_xgboost_loan_model.pkl`: Model trained without log1p transformation

> **Note**: This folder contains binary artifacts, NOT Python source code. Python training code is located in `src/loan_ml/training/`.

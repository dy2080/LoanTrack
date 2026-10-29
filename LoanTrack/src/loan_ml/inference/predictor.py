"""
================================================================================
Inference / Predictor — Save, Load & Run Predictions
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import (
    DEFAULT_MODEL, MODEL_DIR, FEATURE_ORDER, NUMERIC_FEATURES,
)


def save_best_model(model, scaler, label_encoders, model_filename: str = None):
    """Persist the best model + preprocessing artifacts to disk."""
    if model_filename is None:
        model_filename = DEFAULT_MODEL

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": FEATURE_ORDER,
    }
    joblib.dump(artifacts, model_filename)
    print(f"[SUCCESS] Model artifacts saved to '{model_filename}'")


def load_model(model_filename: str = None):
    """Load model + preprocessors from a joblib file."""
    if model_filename is None:
        model_filename = DEFAULT_MODEL
    return joblib.load(model_filename)


def predict_single(raw: dict, model_filename: str = None, apply_log: bool = True):
    """
    Run inference on a single raw applicant dict.

    Parameters:
        raw: dict with keys matching FEATURE_ORDER (raw, human-readable values).
        model_filename: path to the .pkl artifact.
        apply_log: whether log1p was used during training.

    Returns:
        dict with 'status' ('APPROVED' | 'REJECTED') and 'probability' (float 0-1).
    """
    artifacts = load_model(model_filename)
    loaded_model = artifacts["model"]
    scaler = artifacts["scaler"]
    label_encoders = artifacts["label_encoders"]
    feature_names = artifacts["feature_names"]

    sample_df = pd.DataFrame([raw])

    # Encode categoricals
    cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Dependents"]
    for col in cat_cols:
        sample_df[col] = label_encoders[col].transform(sample_df[col].astype(str))

    # Log transform
    if apply_log:
        for col in NUMERIC_FEATURES:
            sample_df[col] = np.log1p(sample_df[col])

    # Scale numerics
    sample_df[NUMERIC_FEATURES] = scaler.transform(sample_df[NUMERIC_FEATURES])

    # Reorder
    sample_df = sample_df[feature_names]

    prediction = loaded_model.predict(sample_df)[0]
    probability = float(loaded_model.predict_proba(sample_df)[0][1])

    return {
        "status": "APPROVED" if prediction == 1 else "REJECTED",
        "probability": probability,
    }


def demo_inference(model_filename: str = None, apply_log: bool = True):
    """Run a demo prediction on a sample applicant and print results."""
    print("\n==========================================================================")
    print("                    INFERENCE DEMO ON NEW SAMPLE APPLICANT")
    print("==========================================================================")

    raw_sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }

    print("Raw Applicant Details:")
    for k, v in raw_sample.items():
        print(f"  {k:<20}: {v}")

    result = predict_single(raw_sample, model_filename, apply_log)

    status_str = f"{result['status']} ({'Y' if result['status'] == 'APPROVED' else 'N'})"
    print(f"\nModel Output:")
    print(f"  Eligibility Status : {status_str}")
    print(f"  Approval Probability: {result['probability'] * 100:.2f}%")

"""
================================================================================
Centralized Configuration for the Loan Eligibility Prediction System
================================================================================
All paths, hyperparameters, feature lists, and constants in one place.
"""

import os

# ---------------------------------------------------------------------------
# Project paths (resolved relative to this file → config/ → project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
VIS_DIR = os.path.join(PROJECT_ROOT, "visualizations")

DEFAULT_DATASET = os.path.join(DATA_DIR, "loan_data.csv")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "xgboost_loan_model.pkl")

# ---------------------------------------------------------------------------
# Remote dataset fallback
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://raw.githubusercontent.com/shrikant-temburwar/"
    "Loan-Prediction-Dataset/master/train.csv"
)

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "Gender", "Married", "Education", "Self_Employed",
    "Property_Area", "Dependents",
]

NUMERIC_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
]

MEDIAN_IMPUTE_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
MODE_IMPUTE_COLS = [
    "Credit_History", "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "Property_Area", "Loan_Amount_Term",
]

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42

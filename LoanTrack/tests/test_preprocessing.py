"""
Tests for the preprocessing pipeline.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.settings import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL, ID_COL
)


def _make_sample_df(n: int = 50) -> pd.DataFrame:
    """Create a small synthetic dataset that mirrors the Kaggle schema."""
    rng = np.random.RandomState(42)
    data = {
        ID_COL: [f"LP{str(i).zfill(4)}" for i in range(n)],
        "Gender": rng.choice(["Male", "Female"], n),
        "Married": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["0", "1", "2", "3+"], n),
        "Education": rng.choice(["Graduate", "Not Graduate"], n),
        "Self_Employed": rng.choice(["Yes", "No"], n),
        "ApplicantIncome": rng.randint(1000, 20000, n).astype(float),
        "CoapplicantIncome": rng.randint(0, 10000, n).astype(float),
        "LoanAmount": rng.randint(50, 500, n).astype(float),
        "Loan_Amount_Term": rng.choice([120, 180, 240, 360], n).astype(float),
        "Credit_History": rng.choice([0.0, 1.0], n),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n),
        TARGET_COL: rng.choice(["Y", "N"], n),
    }
    df = pd.DataFrame(data)
    # Inject a few NaN values
    df.loc[0, "Gender"] = np.nan
    df.loc[1, "LoanAmount"] = np.nan
    df.loc[2, "Credit_History"] = np.nan
    return df


class TestPreprocessing:
    """Unit tests for src.loan_ml.preprocessing.pipeline."""

    def test_no_missing_after_imputation(self):
        from src.loan_ml.preprocessing.pipeline import preprocess_data

        df = _make_sample_df(80)
        X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df, apply_log=False)

        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_output_shapes(self):
        from src.loan_ml.preprocessing.pipeline import preprocess_data

        df = _make_sample_df(100)
        X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df, apply_log=True)

        total = len(X_train) + len(X_test)
        # After balance + split the total may differ from 100, but shapes must be consistent
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_label_encoders_created(self):
        from src.loan_ml.preprocessing.pipeline import preprocess_data

        df = _make_sample_df(80)
        _, _, _, _, _, label_encoders = preprocess_data(df, apply_log=False)

        for col in CATEGORICAL_FEATURES + [TARGET_COL]:
            assert col in label_encoders, f"Missing encoder for '{col}'"

    def test_scaler_fitted(self):
        from src.loan_ml.preprocessing.pipeline import preprocess_data

        df = _make_sample_df(80)
        _, _, _, _, scaler, _ = preprocess_data(df, apply_log=False)

        # StandardScaler should have mean_ attribute after fit
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == len(NUMERIC_FEATURES)

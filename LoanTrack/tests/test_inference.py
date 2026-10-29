"""
Tests for the inference predictor.
"""

import os
import sys
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


SAMPLE_RAW = {
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


class TestInference:
    """Integration tests for save → load → predict_single cycle."""

    @pytest.fixture(autouse=True)
    def _train_and_save(self, tmp_path):
        """Train a quick model and save it to a temp file for each test."""
        from src.loan_ml.data.loader import load_from_url
        from src.loan_ml.preprocessing.pipeline import preprocess_data
        from src.loan_ml.inference.predictor import save_best_model

        import xgboost as xgb

        df = load_from_url()
        X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df, apply_log=True)

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=2, random_state=42, eval_metric="logloss"
        )
        model.fit(X_train, y_train)

        self.model_path = str(tmp_path / "test_model.pkl")
        save_best_model(model, scaler, le, self.model_path)

    def test_predict_returns_dict(self):
        from src.loan_ml.inference.predictor import predict_single

        result = predict_single(SAMPLE_RAW, model_filename=self.model_path, apply_log=True)
        assert isinstance(result, dict)
        assert "status" in result
        assert "probability" in result

    def test_status_values(self):
        from src.loan_ml.inference.predictor import predict_single

        result = predict_single(SAMPLE_RAW, model_filename=self.model_path, apply_log=True)
        assert result["status"] in ("APPROVED", "REJECTED")

    def test_probability_range(self):
        from src.loan_ml.inference.predictor import predict_single

        result = predict_single(SAMPLE_RAW, model_filename=self.model_path, apply_log=True)
        assert 0.0 <= result["probability"] <= 1.0

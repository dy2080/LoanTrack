"""
Models Subpackage — Backward-compatibility layer.
The actual model training and evaluation code has been organized into `src.loan_ml.training`.
"""

from src.loan_ml.training.trainer import build_models, train_all_models
from src.loan_ml.training.evaluator import evaluate_models

__all__ = ["build_models", "train_all_models", "evaluate_models"]

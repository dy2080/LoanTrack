"""
================================================================================
train.py — Quick-start training script
================================================================================
Runs the full pipeline: load → preprocess → train → evaluate → visualize → save
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DATASET_URL, MODEL_DIR, VIS_DIR
from src.loan_ml.data.loader import load_dataset, load_from_url
from src.loan_ml.preprocessing.pipeline import preprocess_data
from src.loan_ml.training.trainer import train_all_models
from src.loan_ml.training.evaluator import evaluate_models
from src.loan_ml.plotting.plots import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_all_roc_curves,
)
from src.loan_ml.inference.predictor import save_best_model, demo_inference


def run_pipeline(use_log: bool = True, tag: str = "default"):
    """Execute one full pipeline run with the given log-transform setting."""
    state_str = "WITH" if use_log else "WITHOUT"
    print(f"\n\n{'*' * 74}")
    print(f"   RUNNING PIPELINE: {state_str} LOG TRANSFORMATION  (tag={tag})")
    print(f"{'*' * 74}")

    # 1. Load
    df = load_from_url()

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(
        df, apply_log=use_log
    )

    # 3. Train
    models = train_all_models(X_train, y_train)

    # 4. Evaluate
    results_df, detailed_metrics = evaluate_models(models, X_test, y_test)

    # 5. Visualizations
    os.makedirs(VIS_DIR, exist_ok=True)
    prefix = f"{tag}_"
    plot_accuracy_comparison(
        results_df,
        output_file=os.path.join(VIS_DIR, f"{prefix}accuracy_comparison.png"),
    )
    plot_confusion_matrix(
        detailed_metrics, y_test,
        best_model_name="XGBoost", prefix=prefix,
    )
    plot_all_roc_curves(
        detailed_metrics, y_test,
        output_file=os.path.join(VIS_DIR, f"{prefix}roc_curves.png"),
    )

    # 6. Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file = os.path.join(MODEL_DIR, f"{prefix}xgboost_loan_model.pkl")
    save_best_model(models["XGBoost"], scaler, label_encoders, model_file)

    # 7. Demo inference
    demo_inference(model_filename=model_file, apply_log=use_log)

    return model_file


def main():
    print("=" * 74)
    print("         END-TO-END LOAN ELIGIBILITY PREDICTION SYSTEM (ML)")
    print("=" * 74)

    # Run both with and without log transformation (like the original script)
    for use_log in [False, True]:
        tag = "with_log" if use_log else "no_log"
        run_pipeline(use_log=use_log, tag=tag)

    print("\n" + "=" * 74)
    print("               PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 74)


if __name__ == "__main__":
    main()

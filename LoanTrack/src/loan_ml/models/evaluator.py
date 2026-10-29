"""
================================================================================
Model Evaluator — Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)


def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluates every trained model on the test set.

    Returns:
        results_df  — pd.DataFrame sorted by accuracy (descending).
        detailed    — dict[name] → metric values + predictions.
    """
    results_list = []
    detailed = {}

    print("\n==========================================================================")
    print("                           MODEL EVALUATION RESULTS")
    print("==========================================================================")

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        cm   = confusion_matrix(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        results_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": auc_val,
        })

        detailed[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc_val,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    results_df = (
        pd.DataFrame(results_list)
        .sort_values(by="Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    # ── Print comparison table ──
    print("\n--- ALGORITHM COMPARISON TABLE ---")
    hdr = (
        f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | "
        f"{'Recall':<10} | {'F1 Score':<10} | {'ROC AUC':<10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, row in results_df.iterrows():
        print(
            f"{row['Model']:<22} | {row['Accuracy']:<10.4f} | "
            f"{row['Precision']:<10.4f} | {row['Recall']:<10.4f} | "
            f"{row['F1 Score']:<10.4f} | {row['ROC AUC']:<10.4f}"
        )

    best_name = results_df.iloc[0]["Model"]
    bm = detailed[best_name]
    print(f"\n[BEST MODEL]: {best_name}")
    print(f"  Accuracy:  {bm['accuracy'] * 100:.2f}%")
    print(f"  Precision: {bm['precision'] * 100:.2f}%")
    print(f"  Recall:    {bm['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {bm['f1_score'] * 100:.2f}%")
    print(f"  ROC AUC:   {bm['roc_auc']:.4f}")
    tn, fp, fn, tp = bm["confusion_matrix"].ravel()
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results_df, detailed

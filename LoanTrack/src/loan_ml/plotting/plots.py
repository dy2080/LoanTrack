"""
================================================================================
Visualization Module — Accuracy bars, Confusion Matrix, ROC Curves
================================================================================
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import VIS_DIR

# Set aesthetic plot style
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
sns.set_theme(style="whitegrid", palette="muted")


def plot_accuracy_comparison(results_df, output_file=None):
    """Bar chart comparing Accuracy across all algorithms."""
    if output_file is None:
        os.makedirs(VIS_DIR, exist_ok=True)
        output_file = os.path.join(VIS_DIR, "model_accuracy_comparison.png")

    plt.figure(figsize=(14, 7))
    colors = [
        "#2ca02c" if m == "XGBoost" else "#1f77b4"
        for m in results_df["Model"]
    ]
    bars = plt.bar(
        results_df["Model"],
        results_df["Accuracy"] * 100,
        color=colors, width=0.55,
        edgecolor="black", linewidth=1.2,
    )

    plt.title(
        f"Algorithm Accuracy Comparison ({len(results_df)} Models)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.xlabel("Machine Learning Algorithms", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.ylim(60, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.8,
            f"{h:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold", rotation=90,
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Accuracy comparison bar chart saved to '{output_file}'")
    plt.close()


def plot_confusion_matrix(detailed_metrics, y_test, best_model_name="XGBoost", prefix=""):
    """Confusion-matrix heatmap for the best model."""
    os.makedirs(VIS_DIR, exist_ok=True)
    cm = detailed_metrics[best_model_name]["confusion_matrix"]

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Rejected (0)", "Approved (1)"],
        yticklabels=["Rejected (0)", "Approved (1)"],
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.title(f"{best_model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    fname = os.path.join(
        VIS_DIR,
        f"{prefix}{best_model_name.lower().replace(' ', '_')}_confusion_matrix.png",
    )
    plt.savefig(fname, dpi=300)
    print(f"[INFO] Confusion matrix saved to '{fname}'")
    plt.close()


def plot_all_roc_curves(detailed_metrics, y_test, output_file=None):
    """Combined ROC curves for every model on one figure."""
    if output_file is None:
        os.makedirs(VIS_DIR, exist_ok=True)
        output_file = os.path.join(VIS_DIR, "roc_curves_comparison.png")

    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap("tab20")

    for idx, (name, metrics) in enumerate(detailed_metrics.items()):
        y_prob = metrics["y_prob"]
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = metrics["roc_auc"]
            if name == "XGBoost":
                color, lw = "#2ca02c", 3
            elif "Voting" in name:
                color, lw = ("#d62728" if "1" in name else "#ff7f0e"), 2.5
            else:
                color, lw = cmap(idx % 20), 1.5
            plt.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Baseline Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    plt.title(
        f"ROC Curves Comparison Across {len(detailed_metrics)} Models",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Combined ROC curves saved to '{output_file}'")
    plt.close()

"""
Visualization Subpackage — Backward-compatibility layer.
The plotting functions have been organized into `src.loan_ml.plotting`.
"""

from src.loan_ml.plotting.plots import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_all_roc_curves,
)

__all__ = [
    "plot_accuracy_comparison",
    "plot_confusion_matrix",
    "plot_all_roc_curves",
]

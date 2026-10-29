"""
================================================================================
Data Loader — Loading & Balancing the Loan Prediction Dataset
================================================================================
"""

import os
import pandas as pd
from sklearn.utils import resample

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import (
    DEFAULT_DATASET, DATASET_URL, TARGET_COL, RANDOM_STATE, DATA_DIR
)


def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Loads the Kaggle Loan Prediction dataset from a local file or raw URL.

    Parameters:
        file_path (str): Path to local CSV file. Falls back to GitHub URL.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    if file_path is None:
        file_path = DEFAULT_DATASET

    if os.path.exists(file_path):
        print(f"[INFO] Loading dataset from local file: '{file_path}'")
        df = pd.read_csv(file_path)
    else:
        print("[INFO] Local file not found. Fetching dataset from GitHub URL...")
        df = pd.read_csv(DATASET_URL)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"[INFO] Dataset saved locally as '{file_path}'")

    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_from_url() -> pd.DataFrame:
    """Always load fresh from the remote URL (used by training pipeline)."""
    print("[INFO] Fetching dataset from GitHub URL...")
    df = pd.read_csv(DATASET_URL)
    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def balance_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Ensures that the target class distribution is balanced 50/50 (Y/N).
    If imbalanced, resamples the minority class to match the majority.
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    print("\n--- Target Class Balance Inspection ---")
    for cls, count in counts.items():
        print(f"  Class '{cls}': {count} samples ({percentages[cls]:.1f}%)")

    max_count = counts.max()
    min_count = counts.min()

    if min_count < max_count:
        print(
            f"[INFO] Target class is imbalanced ({min_count} vs {max_count}). "
            "Resampling minority class to achieve 50/50 split..."
        )
        dfs_balanced = []
        for cls, group in df.groupby(target_col):
            if len(group) < max_count:
                df_resampled = resample(
                    group,
                    replace=True,
                    n_samples=max_count,
                    random_state=random_state,
                )
                dfs_balanced.append(df_resampled)
            else:
                dfs_balanced.append(group)

        balanced_df = (
            pd.concat(dfs_balanced)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        new_counts = balanced_df[target_col].value_counts()
        print(
            f"[SUCCESS] Dataset balanced to 50/50 split "
            f"({new_counts.iloc[0]} vs {new_counts.iloc[1]} samples)."
        )
        return balanced_df
    else:
        print("[SUCCESS] Dataset already has an exact 50/50 balanced split.")
        return df

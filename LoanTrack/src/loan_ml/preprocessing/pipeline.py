"""
================================================================================
Preprocessing Pipeline — Imputation, Encoding, Scaling, Splitting
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    MEDIAN_IMPUTE_COLS, MODE_IMPUTE_COLS,
    TARGET_COL, ID_COL, VIS_DIR,
    TEST_SIZE, RANDOM_STATE,
)
from src.loan_ml.data.loader import balance_dataset


def preprocess_data(df: pd.DataFrame, apply_log: bool = True):
    """
    Executes the full preprocessing pipeline:
      1. Ensure 50/50 target class balance
      2. Missing-value imputation (median / mode)
      3. Log transformation of skewed numeric features (optional)
      4. Label encoding of categorical columns + target
      5. Standard scaling of numeric features
      6. 80:20 stratified train-test split

    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoders
    """
    df_clean = df.copy()

    # 0 — Balance target
    if TARGET_COL in df_clean.columns:
        df_clean = balance_dataset(df_clean)

    # Drop identifier
    if ID_COL in df_clean.columns:
        df_clean = df_clean.drop(columns=[ID_COL])
        print(f"[INFO] Dropped identifier column '{ID_COL}'.")

    # ------------------------------------------------------------------
    # Step 1: Missing Value Imputation
    # ------------------------------------------------------------------
    print("\n--- Preprocessing Step 1: Missing Value Imputation ---")
    print(
        f"Missing values before imputation:\n"
        f"{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}"
    )

    for col in MEDIAN_IMPUTE_COLS:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in MODE_IMPUTE_COLS:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    print(f"Missing values after imputation: {df_clean.isnull().sum().sum()}")

    # ------------------------------------------------------------------
    # Step 2: Log Transformation (optional)
    # ------------------------------------------------------------------
    skewed_features = NUMERIC_FEATURES  # same list
    if apply_log:
        print("\n--- Preprocessing Step 2: Log Transformation & Visualization ---")
        fig, axes = plt.subplots(
            len(skewed_features), 2,
            figsize=(14, 4 * len(skewed_features)),
        )
        fig.suptitle(
            "Distributions Before and After Log Transformation",
            fontsize=16, fontweight="bold",
        )

        for i, col in enumerate(skewed_features):
            sns.histplot(
                df_clean[col].dropna(), kde=True, ax=axes[i, 0],
                color="#1f77b4", stat="density", bins=30,
            )
            axes[i, 0].set_title(f"Before Log ({col})", fontweight="bold")

            df_clean[col] = np.log1p(df_clean[col])

            sns.histplot(
                df_clean[col].dropna(), kde=True, ax=axes[i, 1],
                color="#2ca02c", stat="density", bins=30,
            )
            axes[i, 1].set_title(f"After Log ({col})", fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        os.makedirs(VIS_DIR, exist_ok=True)
        plt.savefig(
            os.path.join(VIS_DIR, "log_transformation_comparison.png"), dpi=300
        )
        plt.close()
        print(f"Applied np.log1p transformation to: {skewed_features}")
    else:
        print("\n--- Preprocessing Step 2: Log Transformation [SKIPPED] ---")

    # ------------------------------------------------------------------
    # Step 3: Label Encoding
    # ------------------------------------------------------------------
    print("\n--- Preprocessing Step 3: Label Encoding ---")
    encode_cols = CATEGORICAL_FEATURES + [TARGET_COL]
    label_encoders = {}
    for col in encode_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
        print(
            f"  Encoded '{col}': "
            f"{dict(zip(le.classes_, le.transform(le.classes_)))}"
        )

    # ------------------------------------------------------------------
    # Step 4: Feature Scaling
    # ------------------------------------------------------------------
    print("\n--- Preprocessing Step 4: Feature Scaling ---")
    scaler = StandardScaler()
    df_clean[NUMERIC_FEATURES] = scaler.fit_transform(
        df_clean[NUMERIC_FEATURES]
    )
    print(f"StandardScaler applied to: {NUMERIC_FEATURES}")

    # ------------------------------------------------------------------
    # Step 5: Stratified Train-Test Split
    # ------------------------------------------------------------------
    print("\n--- Preprocessing Step 5: Stratified Train-Test Split (80:20) ---")
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_clean):.1%})")
    print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(df_clean):.1%})")

    return X_train, X_test, y_train, y_test, scaler, label_encoders

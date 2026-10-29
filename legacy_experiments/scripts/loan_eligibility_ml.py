"""
================================================================================
Loan Eligibility Prediction System - End-to-End ML Pipeline
================================================================================
Dataset: Kaggle "Loan Prediction Dataset" (614 records, 13 features)
Target: Loan_Status (Y = Approved, N = Rejected)

Models Evaluated:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. XGBoost (Best Model)

Preprocessing Pipeline (in exact order):
1. Missing Value Handling: Median imputation for continuous numeric features,
   Mode imputation for binary/categorical features.
2. Label Encoding: LabelEncoder applied to categorical fields.
3. Feature Scaling: StandardScaler applied to continuous numeric features.
4. Train-Test Split: 80:20 stratified split with random_state=42.
================================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)

# Suppress non-critical warnings for clean output
warnings.filterwarnings('ignore')

# Set aesthetic plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")


# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================
def load_dataset(file_path: str = 'loan_data.csv') -> pd.DataFrame:
    """
    Loads the Kaggle Loan Prediction dataset from a local file or raw URL.
    
    Parameters:
        file_path (str): Path to local CSV file or raw dataset URL.
        
    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    raw_url = 'https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv'
    
    if os.path.exists(file_path):
        print(f"[INFO] Loading dataset from local file: '{file_path}'")
        df = pd.read_csv(file_path)
    else:
        print(f"[INFO] Local file not found. Fetching dataset from GitHub URL...")
        df = pd.read_csv(raw_url)
        # Save a local copy for offline execution
        df.to_csv(file_path, index=False)
        print(f"[INFO] Dataset saved locally as '{file_path}'")
        
    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def balance_dataset(df: pd.DataFrame, target_col: str = 'Loan_Status', random_state: int = 42) -> pd.DataFrame:
    """
    Ensures that the target class distribution in the dataset is balanced 50/50 (Y/N).
    If the dataset is imbalanced, resamples the minority class to achieve an exact 50/50 split.
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    print("\n--- Target Class Balance Inspection ---")
    for cls, count in counts.items():
        print(f"  Class '{cls}': {count} samples ({percentages[cls]:.1f}%)")
        
    max_count = counts.max()
    min_count = counts.min()
    
    if min_count < max_count:
        print(f"[INFO] Target class is imbalanced ({min_count} vs {max_count}). Resampling minority class to achieve 50/50 split...")
        dfs_balanced = []
        for cls, group in df.groupby(target_col):
            if len(group) < max_count:
                df_resampled = resample(
                    group,
                    replace=True,
                    n_samples=max_count,
                    random_state=random_state
                )
                dfs_balanced.append(df_resampled)
            else:
                dfs_balanced.append(group)
        balanced_df = pd.concat(dfs_balanced).sample(frac=1, random_state=random_state).reset_index(drop=True)
        new_counts = balanced_df[target_col].value_counts()
        print(f"[SUCCESS] Dataset balanced to 50/50 split ({new_counts.iloc[0]} vs {new_counts.iloc[1]} samples).")
        return balanced_df
    else:
        print("[SUCCESS] Dataset already has an exact 50/50 balanced split.")
        return df


# ==============================================================================
# SECTION 2: DATA PREPROCESSING
# ==============================================================================
def preprocess_data(df: pd.DataFrame, apply_log: bool = True):
    """
    Executes preprocessing pipeline in exact required order:
    1. Check & ensure 50/50 target class balance
    2. Missing value handling (median for numeric, mode for categorical)
    3. Log transformation for skewed numeric features (if apply_log=True)
    4. Label encoding of categorical features & target
    5. Feature scaling using StandardScaler
    6. 80:20 Stratified train-test split
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoders
    """
    df_clean = df.copy()
    
    # 0. Ensure Target Balance (50/50 split)
    if 'Loan_Status' in df_clean.columns:
        df_clean = balance_dataset(df_clean, target_col='Loan_Status')
    
    # Drop identifier column 'Loan_ID' if present
    if 'Loan_ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Loan_ID'])
        print("[INFO] Dropped identifier column 'Loan_ID'.")

    print("\n--- Preprocessing Step 1: Missing Value Imputation ---")
    print(f"Missing values before imputation:\n{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}")

    # Numeric features for median imputation
    median_impute_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in median_impute_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    # Categorical and discrete features for mode imputation
    mode_impute_cols = ['Credit_History', 'Gender', 'Married', 'Dependents', 
                        'Education', 'Self_Employed', 'Property_Area', 'Loan_Amount_Term']
    for col in mode_impute_cols:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)

    print(f"Missing values after imputation: {df_clean.isnull().sum().sum()}")

    if apply_log:
        print("\n--- Preprocessing Step 2: Log Transformation & Visualization ---")
        skewed_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        
        # Create subplots for Before and After comparison
        fig, axes = plt.subplots(len(skewed_features), 2, figsize=(14, 4 * len(skewed_features)))
        fig.suptitle('Distributions Before and After Log Transformation', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(skewed_features):
            # Plot BEFORE transformation
            sns.histplot(df_clean[col].dropna(), kde=True, ax=axes[i, 0], color='#1f77b4', stat='density', bins=30)
            axes[i, 0].set_title(f'Before Log Transformation ({col})', fontweight='bold')
            axes[i, 0].set_xlabel(col)
            
            # Apply transformation
            df_clean[col] = np.log1p(df_clean[col])
            
            # Plot AFTER transformation
            sns.histplot(df_clean[col].dropna(), kde=True, ax=axes[i, 1], color='#2ca02c', stat='density', bins=30)
            axes[i, 1].set_title(f'After Log Transformation ({col})', fontweight='bold')
            axes[i, 1].set_xlabel(f'{col}_Log')

        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout so suptitle fits
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/log_transformation_comparison.png', dpi=300)
        plt.close()
        
        print(f"Applied np.log1p transformation to: {skewed_features}")
        print("[INFO] Log transformation before/after graph saved to 'visualizations/log_transformation_comparison.png'")
    else:
        print("\n--- Preprocessing Step 2: Log Transformation [SKIPPED] ---")

    print("\n--- Preprocessing Step 3: Label Encoding ---")
    categorical_fields = ['Gender', 'Married', 'Education', 'Self_Employed', 
                          'Property_Area', 'Dependents', 'Loan_Status']
    label_encoders = {}
    
    for col in categorical_fields:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    print("\n--- Preprocessing Step 4: Feature Scaling ---")
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    scaler = StandardScaler()
    df_clean[numeric_features] = scaler.fit_transform(df_clean[numeric_features])
    print(f"StandardScaler applied to: {numeric_features}")

    print("\n--- Preprocessing Step 5: Stratified Train-Test Split (80:20) ---")
    X = df_clean.drop(columns=['Loan_Status'])
    y = df_clean['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df_clean):.1%})")
    print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(df_clean):.1%})")
    print(f"Target distribution in Training set: Rejected (N={np.sum(y_train==0)}, {np.sum(y_train==0)/len(y_train):.1%}), Approved (Y={np.sum(y_train==1)}, {np.sum(y_train==1)/len(y_train):.1%})")
    print(f"Target distribution in Test set:     Rejected (N={np.sum(y_test==0)}, {np.sum(y_test==0)/len(y_test):.1%}), Approved (Y={np.sum(y_test==1)}, {np.sum(y_test==1)/len(y_test):.1%})")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders


# ==============================================================================
# SECTION 3: MODEL TRAINING
# ==============================================================================
def train_all_models(X_train, y_train):
    """
    Trains various machine learning models including new ensemble and neural network methods.
    
    Returns:
        dict: Mapping of model name to trained model object.
    """
    models = {
        'Logistic Regression': LogisticRegression(C=1.0, solver='lbfgs', random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=2, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=42),
        'SVM': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'KNeighbors': KNeighborsClassifier(n_neighbors=5),
        'GaussianNB': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'Dense Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1, random_state=42, eval_metric='logloss'
        )
    }

    # Voting Ensemble 1 (All Models - Soft Voting)
    # Exclude models that don't support predict_proba inherently without specific configurations, 
    # though all selected here do support it (SVC has probability=True)
    all_estimators = [(name, model) for name, model in models.items()]
    models['Voting Ensemble 1'] = VotingClassifier(estimators=all_estimators, voting='soft')

    # Voting Ensemble 2 (Best Three Models - Soft Voting)
    # Using XGBoost, Gradient Boosting, and Random Forest as the typical top 3 for tabular data
    best_3_estimators = [
        ('XGBoost', models['XGBoost']),
        ('Gradient Boosting', models['Gradient Boosting']),
        ('Random Forest', models['Random Forest'])
    ]
    models['Voting Ensemble 2'] = VotingClassifier(estimators=best_3_estimators, voting='soft')

    print("\n==========================================================================")
    print(f"               TRAINING {len(models)} MACHINE LEARNING MODELS                  ")
    print("==========================================================================")
    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)

    print(f"[SUCCESS] All {len(models)} models trained successfully.")
    return models


# ==============================================================================
# SECTION 4: MODEL EVALUATION
# ==============================================================================
def evaluate_models(models, X_test, y_test):
    """
    Evaluates all trained models on the test set and calculates:
    Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC AUC.
    
    Returns:
        pd.DataFrame: Summary table of evaluation metrics.
        dict: Detailed results per model.
    """
    results_list = []
    detailed_metrics = {}

    print("\n==========================================================================")
    print("                           MODEL EVALUATION RESULTS                       ")
    print("==========================================================================")

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        results_list.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC AUC': auc_val
        })

        detailed_metrics[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc_val,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    # Print clean comparison table
    print("\n--- ALGORITHM COMPARISON TABLE ---")
    hdr = f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'ROC AUC':<10}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<22} | {row['Accuracy']:<10.4f} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | {row['F1 Score']:<10.4f} | {row['ROC AUC']:<10.4f}")

    # Best Model highlight
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n[BEST MODEL]: {best_model_name}")
    bm_metrics = detailed_metrics[best_model_name]
    print(f"  Accuracy:  {bm_metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {bm_metrics['precision'] * 100:.2f}%")
    print(f"  Recall:    {bm_metrics['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {bm_metrics['f1_score'] * 100:.2f}%")
    print(f"  ROC AUC:   {bm_metrics['roc_auc']:.4f}")
    cm_ravel = bm_metrics['confusion_matrix'].ravel()
    print(f"  Confusion Matrix: TN={cm_ravel[0]}, FP={cm_ravel[1]}, FN={cm_ravel[2]}, TP={cm_ravel[3]}")

    return results_df, detailed_metrics


# ==============================================================================
# SECTION 5: VISUALIZATIONS
# ==============================================================================
def plot_accuracy_comparison(results_df, output_file='model_accuracy_comparison.png'):
    """
    Generates and saves a bar chart comparing Accuracy across all algorithms.
    """
    plt.figure(figsize=(14, 7))
    colors = ['#2ca02c' if m == 'XGBoost' else '#1f77b4' for m in results_df['Model']]
    
    bars = plt.bar(results_df['Model'], results_df['Accuracy'] * 100, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    
    plt.title(f'Algorithm Accuracy Comparison ({len(results_df)} Models)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Machine Learning Algorithms', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.ylim(60, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')

    # Add accuracy text labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.8,
            f"{height:.1f}%",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            rotation=90
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Accuracy comparison bar chart saved to '{output_file}'")
    plt.close()


def plot_best_model_details(detailed_metrics, y_test, best_model_name='XGBoost', prefix=''):
    """
    Plots the Confusion Matrix heat map for the specified best performing model (XGBoost).
    """
    metrics = detailed_metrics[best_model_name]
    cm = metrics['confusion_matrix']
    
    plt.figure(figsize=(7, 5.5))

    # Confusion Matrix Heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=['Rejected (0)', 'Approved (1)'],
        yticklabels=['Rejected (0)', 'Approved (1)'],
        annot_kws={"size": 16, "weight": "bold"}
    )
    plt.title(f'{best_model_name} - Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')

    plt.tight_layout()
    cm_roc_file = f'{prefix}{best_model_name.lower().replace(" ", "_")}_confusion_matrix.png'
    plt.savefig(cm_roc_file, dpi=300)
    print(f"[INFO] Best model ({best_model_name}) Confusion Matrix saved to '{cm_roc_file}'")
    plt.close()


def plot_all_roc_curves(detailed_metrics, y_test, output_file='roc_curves_comparison.png'):
    """
    Plots combined ROC curves for all models on a single figure.
    """
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20')
    
    for idx, (name, metrics) in enumerate(detailed_metrics.items()):
        y_prob = metrics['y_prob']
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = metrics['roc_auc']
            # Highlight XGBoost and Ensembles
            if name == 'XGBoost':
                color, lw = '#2ca02c', 3
            elif 'Voting' in name:
                color, lw = '#d62728' if '1' in name else '#ff7f0e', 2.5
            else:
                color, lw = cmap(idx % 20), 1.5
            
            plt.plot(fpr, tpr, color=color, lw=lw, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--', label='Baseline Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curves Comparison Across {len(detailed_metrics)} Models', fontsize=14, fontweight='bold', pad=15)
    
    # Place legend outside to avoid covering curves
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[INFO] Combined ROC curves comparison saved to '{output_file}'")
    plt.close()


# ==============================================================================
# SECTION 6: MODEL PERSISTENCE & INFERENCE DEMO
# ==============================================================================
def save_best_model(model, scaler, label_encoders, model_filename='xgboost_loan_model.pkl'):
    """
    Saves the best model and artifact preprocessors using joblib for inference.
    """
    artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
    }
    joblib.dump(artifacts, model_filename)
    print(f"[SUCCESS] Trained best model and preprocessing artifacts saved to '{model_filename}'")


def demo_inference(model_filename='xgboost_loan_model.pkl', apply_log=True):
    """
    Demonstrates model loading and inference on a new sample applicant profile.
    """
    print("\n==========================================================================")
    print("                    INFERENCE DEMO ON NEW SAMPLE APPLICANT                ")
    print("==========================================================================")
    
    artifacts = joblib.load(model_filename)
    loaded_model = artifacts['model']
    scaler = artifacts['scaler']
    label_encoders = artifacts['label_encoders']
    feature_names = artifacts['feature_names']

    # Sample applicant payload
    raw_sample = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }

    sample_df = pd.DataFrame([raw_sample])
    print("Raw Applicant Details:")
    for k, v in raw_sample.items():
        print(f"  {k:<20}: {v}")

    # Transform categorical features
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
        sample_df[col] = label_encoders[col].transform(sample_df[col].astype(str))

    if apply_log:
        # Apply log transformation to match the preprocessing pipeline
        skewed_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        for col in skewed_features:
            sample_df[col] = np.log1p(sample_df[col])

    # Scale numeric features
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    sample_df[numeric_features] = scaler.transform(sample_df[numeric_features])

    # Reorder columns to match training set
    sample_df = sample_df[feature_names]

    # Predict
    prediction = loaded_model.predict(sample_df)[0]
    probability = loaded_model.predict_proba(sample_df)[0][1]

    status_str = "APPROVED (Y)" if prediction == 1 else "REJECTED (N)"
    print(f"\nModel Output:")
    print(f"  Eligibility Status : {status_str}")
    print(f"  Approval Probability: {probability * 100:.2f}%")


# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
def main():
    print("==========================================================================")
    print("         END-TO-END LOAN ELIGIBILITY PREDICTION SYSTEM (ML)              ")
    print("==========================================================================")
    
    # Use ONLY the original Kaggle dataset
    data_source = 'https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv'

    for use_log in [False, True]:
        state_str = "AFTER (WITH)" if use_log else "BEFORE (WITHOUT)"
        print(f"\n\n**************************************************************************")
        print(f"               RUNNING PIPELINE: {state_str} LOG TRANSFORMATION")
        print(f"**************************************************************************")
        
        # 1. Load Data
        df = pd.read_csv(data_source)
        print("[INFO] Loaded Original Kaggle Dataset from URL.")
        print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # 2. Preprocess Data
        X_train, X_test, y_train, y_test, scaler, label_encoders = preprocess_data(df, apply_log=use_log)

        # 3. Train Models
        models = train_all_models(X_train, y_train)

        # 4. Evaluate Models
        results_df, detailed_metrics = evaluate_models(models, X_test, y_test)

        # 5. Generate Visualizations
        os.makedirs('visualizations', exist_ok=True)
        prefix_mod = "with_log" if use_log else "no_log"
        img_prefix = f'visualizations/original_{prefix_mod}_'
        model_prefix = f'original_{prefix_mod}_'
        
        plot_accuracy_comparison(results_df, output_file=f'{img_prefix}model_accuracy_comparison.png')
        plot_best_model_details(detailed_metrics, y_test, best_model_name='XGBoost', prefix=img_prefix)
        plot_all_roc_curves(detailed_metrics, y_test, output_file=f'{img_prefix}roc_curves_comparison.png')

        # 6. Save Best Model (XGBoost)
        best_model = models['XGBoost']
        save_best_model(best_model, scaler, label_encoders, model_filename=f'{model_prefix}xgboost_loan_model.pkl')

        # 7. Demo Inference
        demo_inference(model_filename=f'{model_prefix}xgboost_loan_model.pkl', apply_log=use_log)

    print("\n==========================================================================")
    print("               PIPELINE EXECUTION COMPLETED SUCCESSFULLY                  ")
    print("==========================================================================")

if __name__ == '__main__':
    main()

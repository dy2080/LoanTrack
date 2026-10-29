import pandas as pd
import numpy as np

def compare_datasets():
    print("=========================================================")
    print("       DATASET COMPARISON: ORIGINAL VS SYNTHETIC         ")
    print("=========================================================\n")
    
    # Load synthetic dataset
    try:
        synthetic_df = pd.read_csv('loan_data.csv')
    except Exception as e:
        print(f"Error loading synthetic dataset: {e}")
        return

    # Load original dataset (using the GitHub raw link which mirrors the Kaggle data)
    print("Fetching original Kaggle dataset...")
    original_url = 'https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv'
    try:
        original_df = pd.read_csv(original_url)
    except Exception as e:
        print(f"Error fetching original dataset: {e}")
        return

    # 1. Shape Comparison
    print("\n--- 1. Dataset Shape ---")
    print(f"Original Dataset : {original_df.shape[0]} rows, {original_df.shape[1]} columns")
    print(f"Synthetic Dataset: {synthetic_df.shape[0]} rows, {synthetic_df.shape[1]} columns")
    
    # 2. Missing Values Comparison
    print("\n--- 2. Missing Values (Total per dataset) ---")
    print(f"Original Dataset : {original_df.isnull().sum().sum()} total missing values")
    print(f"Synthetic Dataset: {synthetic_df.isnull().sum().sum()} total missing values")
    
    # 3. Target Variable Distribution
    print("\n--- 3. Target Variable Distribution ('Loan_Status') ---")
    orig_target = original_df['Loan_Status'].value_counts(normalize=True) * 100
    synth_target = synthetic_df['Loan_Status'].value_counts(normalize=True) * 100
    
    print("Original Dataset:")
    print(f"  Approved (Y): {orig_target.get('Y', 0):.1f}%")
    print(f"  Rejected (N): {orig_target.get('N', 0):.1f}%")
    
    print("\nSynthetic Dataset:")
    print(f"  Approved (Y): {synth_target.get('Y', 0):.1f}%")
    print(f"  Rejected (N): {synth_target.get('N', 0):.1f}%")

    # 4. Numeric Feature Summary
    print("\n--- 4. Numeric Feature Averages ---")
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    
    for col in numeric_cols:
        orig_mean = original_df[col].mean()
        synth_mean = synthetic_df[col].mean()
        print(f"Feature: {col}")
        print(f"  Original Mean : {orig_mean:.2f}")
        print(f"  Synthetic Mean: {synth_mean:.2f}")

    print("\n=========================================================")
    print("                 COMPARISON COMPLETE                     ")
    print("=========================================================")

if __name__ == '__main__':
    compare_datasets()

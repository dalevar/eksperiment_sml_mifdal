"""
automate_mifdal.py
Automated preprocessing pipeline for Heart Disease dataset
Author: mifdal
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(csv_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path} | Shape: {df.shape}")
    return df

def handle_missing_values(df):
    """Fill missing values: median for numeric, mode for categorical."""
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()

def cap_outliers(df, exclude_cols=[]):
    """Cap outliers using IQR method for numeric columns."""
    num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in exclude_cols]
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def encode_categorical(df, exclude_cols=[]):
    """Encode categorical columns: label for binary, one-hot for multi-class."""
    le = LabelEncoder()
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in exclude_cols]
    for col in cat_cols:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    return df

def scale_features(X):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def split_save_data(X, y, out_dir, prefix="heart_disease"): 
    """Split data, save preprocessed, train, and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs(out_dir, exist_ok=True)
    pd.concat([X, y], axis=1).to_csv(os.path.join(out_dir, f"{prefix}_preprocessed.csv"), index=False)
    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(out_dir, "train_data.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(out_dir, "test_data.csv"), index=False)
    print(f"Saved preprocessed, train, and test data to {out_dir}")
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(csv_path, out_dir, target_candidates=None):
    """Full preprocessing pipeline."""
    if target_candidates is None:
        target_candidates = ['target', 'disease', 'condition', 'heart_disease', 'output', 'HeartDisease']
    df = load_dataset(csv_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    # Identify target column
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    # Outlier capping (exclude target)
    df = cap_outliers(df, exclude_cols=[target_col])
    # Encode categorical (exclude target)
    df = encode_categorical(df, exclude_cols=[target_col])
    # Split X/y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Feature scaling
    X_scaled = scale_features(X)
    # Split and save
    return split_save_data(X_scaled, y, out_dir, prefix="heart_disease")

if __name__ == "__main__":
    # Example usage
    csv_path = os.path.join("dataset_raw", "Heart_Disease_Prediction.csv")
    out_dir = os.path.join("preprocessing", "heart_disease_preprocessed")
    preprocess_pipeline(csv_path, out_dir)

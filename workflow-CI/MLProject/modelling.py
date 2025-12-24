import argparse
import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main(train_data, test_data):
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)
    target_col = train_df.columns[-1]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="LogisticRegression_CI"):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    main(args.train_data, args.test_data)

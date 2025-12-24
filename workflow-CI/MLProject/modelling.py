import argparse
import os
from xml.parsers.expat import model
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

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
    # mlflow.sklearn.autolog()  # Disabled to avoid autolog overriding manual log_model

    with mlflow.start_run(run_name="LogisticRegression_CI") as run:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        print("Logging model...")
        mlflow.sklearn.log_model(model, "model", input_example=X_test[:2], signature=mlflow.models.infer_signature(X_test, y_pred))
        print("Model logged.")
        print("MLflow run_id:", run.info.run_id)
        artifact_uri = mlflow.get_artifact_uri("model")
        print(f"Model artifact saved at: {artifact_uri}")
        # List isi artifacts untuk debug
        artifacts_dir = os.path.join("mlruns", "0", run.info.run_id, "artifacts")
        if os.path.exists(artifacts_dir):
            print("Artifacts dir:", os.listdir(artifacts_dir))
            model_dir = os.path.join(artifacts_dir, "model")
            if os.path.exists(model_dir):
                print("Model dir exists:", os.listdir(model_dir))
            else:
                print("Model dir does NOT exist in artifacts!")
        else:
            print("Artifacts dir does NOT exist!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    main(args.train_data, args.test_data)

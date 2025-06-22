import argparse
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
import mlflow
import sys
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Hapus set_tracking_uri dan set_experiment

    try:
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("random_state", args.random_state)

        # Load dataset
        df = pd.read_csv("valve_plate_clean.csv")
        X = df.drop(columns=["label", "stan", "Czas", "Czas2"], errors='ignore')
        y = df["label"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=args.random_state
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, random_state=args.random_state
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        loss = log_loss(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("log_loss", loss)

        # Save and log model
        os.makedirs("rf_model", exist_ok=True)
        joblib.dump(model, "rf_model/model.pkl")
        mlflow.log_artifacts("rf_model")

    except Exception as e:
        mlflow.log_param("error", str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
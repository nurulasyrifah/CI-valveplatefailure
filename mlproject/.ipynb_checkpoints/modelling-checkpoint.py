import pandas as pd
import numpy as np
import os
import shutil
import joblib
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss


import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ===============================
# MLflow Setup
# ===============================
mlflow.set_tracking_uri("https://dagshub.com/nurulasyrifah/valveplate-failure-detection.mlflow")

# Autentikasi ke DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "nurulasyrifah"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "48f229c40dc83870748a2517d33c7539df94db26"

# Nama eksperimen
experiment_name = "valveplate-failure-detection"
client = MlflowClient()

# ===============================
# Cek dan Set Experiment
# ===============================
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog()

print(f"✅ Experiment ID: {experiment_id}")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("valve_plate_clean.csv")

# Drop kolom yang tidak digunakan
X = df.drop(columns=["label", "stan", "Czas", "Czas2"], errors='ignore')
y = df["label"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Start MLflow Run
# ===============================
with mlflow.start_run(run_name="RandomForest Run"):
    # Tag tambahan
    mlflow.set_tag("author", "Nurul Asyrifah")
    mlflow.set_tag("project", "Valve Plate Failure Detection")

    # Logging parameter model
    #n_estimators = 100
    #random_state = 42
    #mlflow.log_param("n_estimators", n_estimators)
    #mlflow.log_param("random_state", random_state)

    # Logging hash dataset
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    mlflow.log_param("data_hash", data_hash)

    # ===========================
    # Model Training
    # ===========================
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ===========================
    # Evaluation
    # ===========================
    #y_pred = model.predict(X_test)
    #y_proba = model.predict_proba(X_test)

    #acc = accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred, average="weighted")
    #loss = log_loss(y_test, y_proba)

    #mlflow.log_metric("accuracy", acc)
    #mlflow.log_metric("f1_score", f1)
    #lflow.log_metric("log_loss", loss)

    # ===========================
    # Save & Log Model
    # ===========================
    model_path = "rf_model"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, os.path.join(model_path, "model.pkl"))
    mlflow.log_artifacts(model_path)

    print("✅ Run berhasil dan sudah dicatat di MLflow DagsHub.")

# ===============================
# Jalankan dari Terminal
# ===============================
if __name__ == "__main__":
    print("Memulai training dan pencatatan ke MLflow...")

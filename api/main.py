from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.xgboost
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from src.utils.config import Config
from src.features.engineering import FeatureEngineer
from src.features.feature_store import FeatureStore

app = FastAPI(title="Fraud Detection API")
config = Config()

# Artifact cache
model = None
fe = None
fs = FeatureStore(window_hours=24)

class Transaction(BaseModel):
    user_id: int
    signup_time: str
    purchase_time: str
    purchase_value: float
    device_id: str
    source: str
    browser: str
    sex: str
    age: int
    ip_address: int

@app.on_event("startup")
def load_artifacts() -> None:
    """Loads model and feature engineering artifacts on API startup."""
    global model, fe
    try:
        # Try to load from MLflow if tracking URI is reachable
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        # For simplicity, we load the latest run from the experiment
        experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
            if not runs.empty:
                latest_run_id = runs.iloc[0].run_id
                model_uri = f"runs:/{latest_run_id}/model"
                model = mlflow.xgboost.load_model(model_uri)
                print(f"Loaded model from MLflow run: {latest_run_id}")
        
        # Load preprocessor
        if os.path.exists("models/feature_engineer.joblib"):
            fe = joblib.load("models/feature_engineer.joblib")
            print("Loaded feature engineer from local storage")
    except Exception as e:
        print(f"Warning: Could not load artifacts from MLflow: {e}")
        # Fallback logic could go here

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Returns the health status of the API and loaded artifacts."""
    return {"status": "healthy", "model_loaded": model is not None, "preprocessor_loaded": fe is not None}

@app.post("/predict")
def predict(tx: Transaction) -> Dict[str, Any]:
    """
    Receives transaction data, calculates real-time velocity, and predicts fraud risk.
    
    Args:
        tx (Transaction): Transaction data in JSON format.
        
    Returns:
        Dict: Fraud probability, binary prediction, and risk level.
    """
    if model is None or fe is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    # 1. Convert to DataFrame and get velocity
    data = pd.DataFrame([tx.dict()])
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    
    # Update real-time feature store
    velocity = fs.update_and_get_velocity(tx.user_id, data['purchase_time'].iloc[0])
    
    # 2. Preprocess with velocity override
    try:
        X = fe.transform(data, is_training=False, velocity_override=velocity)
        
        # 3. Predict
        proba = model.predict_proba(X)[0][1]
        prediction = int(proba > 0.5)

        # 4. Log inference for drift detection
        log_file = "data-set/inference_logs.csv"
        # We log the raw features + some engineered ones if needed, 
        # but for drift we mostly care about inputs and eventually outputs.
        log_entry = data.copy()
        log_entry['fraud_probability'] = float(proba)
        log_entry['prediction'] = prediction
        
        if not os.path.exists(log_file):
            log_entry.to_csv(log_file, index=False)
        else:
            log_entry.to_csv(log_file, mode='a', header=False, index=False)
        
        return {
            "fraud_probability": float(proba),
            "prediction": prediction,
            "risk_level": "High" if proba > 0.8 else "Medium" if proba > 0.5 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from src.utils.config import Config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer
import joblib

def train_model():
    config = Config()
    
    # 1. Load Data
    loader = DataLoader(config)
    df_fraud = loader.load_fraud_data()
    ip_map = loader.load_ip_country_map()
    
    # 2. Feature Engineering
    fe = FeatureEngineer(config)
    fe.fit_ip_map(ip_map)
    
    print("Engineering features...")
    df_transformed = fe.transform(df_fraud, is_training=True)
    
    # Define features and target
    X = df_transformed.drop('class', axis=1)
    y = df_transformed['class']
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # 4. Handle Imbalance (SMOTE on training set only)
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy=0.5, random_state=config.RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 5. MLflow Tracking
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        print("Training model...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='aucpr'
        )
        
        model.fit(X_train_res, y_train_res)
        
        # 6. Evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        
        print(f"Model trained. PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}")
        
        # 7. SHAP Explainability
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Global Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()
        
        # 8. Save Model and Preprocessors
        mlflow.xgboost.log_model(model, "model")
        
        # Save preprocessors separately for inference
        os.makedirs("models", exist_ok=True)
        joblib.dump(fe, "models/feature_engineer.joblib")
        mlflow.log_artifact("models/feature_engineer.joblib")

if __name__ == "__main__":
    train_model()

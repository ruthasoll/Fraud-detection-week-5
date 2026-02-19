from dataclasses import dataclass
import os

@dataclass
class Config:
    """Project-wide configuration settings using dataclasses."""
    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = "Fraud_Detection_Week_12"
    
    # Data paths
    RAW_DATA_PATH: str = "data-set/raw"
    PROCESSED_DATA_PATH: str = "data-set/processed"
    
    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # Feature paths
    IP_TO_COUNTRY_PATH: str = "data-set/raw/IpAddress_to_Country.csv"
    FRAUD_DATA_PATH: str = "data-set/raw/Fraud_Data.csv"
    CREDIT_CARD_PATH: str = "data-set/raw/creditcard.csv"

# Predefined constants
FRAUD_COLORS = ["#1a73e8", "#d93025"]  # Blue for legit, Red for fraud

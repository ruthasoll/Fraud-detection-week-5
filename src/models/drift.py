import pandas as pd
import os
from src.utils.config import Config
from src.data.loader import DataLoader
from src.utils.monitoring import Monitor

def check_drift() -> None:
    """
    Orchestrates the data drift analysis process.
    Loads reference and current data, aligns features, and generates an HTML report.
    """
    config = Config()
    loader = DataLoader(config)
    
    print("Loading reference data (training set)...")
    # In a real scenario, you'd use a specific versioned training split
    # For this challenge, we'll use the raw fraud data as baseline
    ref_df = loader.load_fraud_data()
    
    inference_log_path = "data-set/inference_logs.csv"
    if not os.path.exists(inference_log_path):
        print(f"Error: Inference logs not found at {inference_log_path}. Send some transactions first!")
        return

    print("Loading current data (inference logs)...")
    curr_df = pd.read_csv(inference_log_path)
    
    # 1. Align columns: use only features present in both
    features = ['purchase_value', 'source', 'browser', 'sex', 'age', 'ip_address']
    
    ref_df = ref_df[features]
    curr_df = curr_df[features]
    
    # Simple type alignment
    for col in features:
        curr_df[col] = curr_df[col].astype(ref_df[col].dtype)

    monitor = Monitor(reference_data=ref_df)
    
    print("Running drift analysis...")
    monitor.run_drift_check(current_data=curr_df, output_path="docs/drift_report.html")
    
    print("Drift detection complete. Report saved to docs/drift_report.html")

if __name__ == "__main__":
    check_drift()

import pandas as pd
from evidently import Report, metrics
from evidently.presets import DataDriftPreset
import os

class Monitor:
    """Class for monitoring model and data health using Evidently AI."""
    
    def __init__(self, reference_data: pd.DataFrame):
        """Initializes the monitor with a reference dataset (usually training data)."""
        self.reference_data = reference_data

    def run_drift_check(self, current_data: pd.DataFrame, output_path: str = "docs/drift_report.html"):
        """
        Generates an Evidently data drift report comparing reference and current data.
        
        Args:
            current_data (pd.DataFrame): The production data to check for drift.
            output_path (str): The local path where the HTML report should be saved.
            
        Returns:
            Snapshot: The generated Evidently snapshot object.
        """
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        snapshot = report.run(reference_data=self.reference_data, current_data=current_data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        snapshot.save_html(output_path)
        print(f"Drift report saved to {output_path}")
        return snapshot

    @staticmethod
    def check_quality(df: pd.DataFrame) -> bool:
        """
        Performs basic data quality checks such as checking for null values.
        
        Returns:
            bool: True if data passes all quality checks, False otherwise.
        """
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            print(f"Warning: Data contains {nulls} null values.")
        return nulls == 0

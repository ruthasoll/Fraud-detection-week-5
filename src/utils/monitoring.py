import pandas as pd
from evidently import Report, metrics
from evidently.presets import DataDriftPreset
import os

class Monitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data

    def run_drift_check(self, current_data: pd.DataFrame, output_path: str = "docs/drift_report.html"):
        """Generates an Evidently drift report."""
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        snapshot = report.run(reference_data=self.reference_data, current_data=current_data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        snapshot.save_html(output_path)
        print(f"Drift report saved to {output_path}")
        return snapshot

    @staticmethod
    def check_quality(df: pd.DataFrame):
        """Simple data quality check."""
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            print(f"Warning: Data contains {nulls} null values.")
        return nulls == 0

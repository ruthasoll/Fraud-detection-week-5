import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import random
from src.utils.config import Config
from src.data.loader import DataLoader

class TransactionSimulator:
    def __init__(self, api_url: str = "http://localhost:8000/predict"):
        self.api_url = api_url
        self.config = Config()
        self.loader = DataLoader(self.config)
        self.df_raw = self.loader.load_fraud_data()
        
        # In-memory state for windowed features (transaction velocity)
        # key: user_id, value: list of timestamps of recent transactions
        self.user_state = {}

    def get_velocity(self, user_id: int, window_minutes: int = 1440) -> int:
        """Calculates transaction velocity in the last X minutes."""
        now = datetime.now()
        if user_id not in self.user_state:
            return 0
        
        # Filter timestamps within window
        cutoff = now - timedelta(minutes=window_minutes)
        self.user_state[user_id] = [ts for ts in self.user_state[user_id] if ts > cutoff]
        
        return len(self.user_state[user_id])

    def send_transaction(self):
        """Sends a random transaction from the dataset to the API."""
        row = self.df_raw.sample(n=1).iloc[0]
        
        # Update state (simulated 'now')
        user_id = int(row['user_id'])
        now = datetime.now()
        if user_id not in self.user_state:
            self.user_state[user_id] = []
        self.user_state[user_id].append(now)
        
        # Prepare payload
        payload = {
            "user_id": int(row['user_id']),
            "signup_time": str(row['signup_time']),
            "purchase_time": str(now), # Use current time for simulation
            "purchase_value": float(row['purchase_value']),
            "device_id": str(row['device_id']),
            "source": str(row['source']),
            "browser": str(row['browser']),
            "sex": str(row['sex']),
            "age": int(row['age']),
            "ip_address": int(row['ip_address'])
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            print(f"User {user_id} | Prob: {response.json().get('fraud_probability'):.3f} | Prediction: {response.json().get('prediction')}")
            return response.json()
        except Exception as e:
            print(f"Failed to send transaction: {e}")
            return None

    def run(self, num_tx: int = 10, delay: float = 1.0):
        print(f"Starting simulation. Sending {num_tx} transactions to {self.api_url}...")
        for i in range(num_tx):
            self.send_transaction()
            time.sleep(delay)

if __name__ == "__main__":
    simulator = TransactionSimulator()
    simulator.run(num_tx=20, delay=0.5)

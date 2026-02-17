from datetime import datetime, timedelta
from typing import Dict, List

class FeatureStore:
    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        # Cache for transaction timestamps per user
        self.user_tx_history: Dict[int, List[datetime]] = {}

    def update_and_get_velocity(self, user_id: int, current_time: datetime) -> int:
        """Adds a new transaction timestamp and returns the count within the window."""
        if user_id not in self.user_tx_history:
            self.user_tx_history[user_id] = []
        
        # Add current transaction
        self.user_tx_history[user_id].append(current_time)
        
        # Clean up old timestamps
        cutoff = current_time - timedelta(hours=self.window_hours)
        self.user_tx_history[user_id] = [
            ts for ts in self.user_tx_history[user_id] if ts > cutoff
        ]
        
        return len(self.user_tx_history[user_id])

    def reset(self):
        self.user_tx_history = {}

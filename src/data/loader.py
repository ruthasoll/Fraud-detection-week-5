import pandas as pd
import numpy as np
from typing import Tuple, Dict
from src.utils.config import Config

class DataLoader:
    """Utility class for loading project datasets."""
    
    def __init__(self, config: Config):
        """Initializes the DataLoader with project configuration."""
        self.config = config

    def load_fraud_data(self) -> pd.DataFrame:
        """Loads the raw fraud dataset and converts time columns to datetime objects."""
        df = pd.read_csv(self.config.FRAUD_DATA_PATH)
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        return df

    def load_credit_card_data(self) -> pd.DataFrame:
        """Loads the raw credit card dataset from the configured path."""
        return pd.read_csv(self.config.CREDIT_CARD_PATH)

    def load_ip_country_map(self) -> pd.DataFrame:
        """Loads and prepares the IP-to-Country mapping dataset."""
        df_ip = pd.read_csv(self.config.IP_TO_COUNTRY_PATH)
        df_ip['lower_bound_ip_address'] = df_ip['lower_bound_ip_address'].astype('int64')
        df_ip['upper_bound_ip_address'] = df_ip['upper_bound_ip_address'].astype('int64')
        return df_ip

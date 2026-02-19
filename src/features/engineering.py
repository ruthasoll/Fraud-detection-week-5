import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple, Optional
from src.utils.config import Config

class FeatureEngineer:
    """Handles feature engineering, scaling, and encoding for fraud detection."""
    
    def __init__(self, config: Config):
        """Initializes the FeatureEngineer with a configuration object."""
        self.config = config
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.ip_map: Optional[pd.DataFrame] = None

    def fit_ip_map(self, ip_map: pd.DataFrame) -> None:
        """Sets the IP-to-country mapping dataframe."""
        self.ip_map = ip_map

    def get_country(self, ip: int) -> str:
        """Maps an integer IP address to a country string."""
        if self.ip_map is None:
            return "Unknown"
        match = self.ip_map[
            (self.ip_map['lower_bound_ip_address'] <= ip) &
            (self.ip_map['upper_bound_ip_address'] >= ip)
        ]
        return str(match['country'].iloc[0]) if not match.empty else "Unknown"

    def engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates time-based features from signup and purchase timestamps."""
        df = df.copy()
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup_hours'] = (
            (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        )
        return df

    def calculate_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates transaction count for each user in the last 24 hours."""
        df_sorted = df.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)
        df_indexed = df_sorted.set_index('purchase_time')
        
        velocity_series = df_indexed.groupby('user_id', group_keys=False).apply(
            lambda group: group['user_id'].rolling(
                window='24h',
                min_periods=1,
                closed='both'
            ).count()
        ).reset_index(drop=True)
        
        df_sorted['tx_count_last_24h'] = velocity_series
        return df_sorted

    def transform(self, df: pd.DataFrame, is_training: bool = False, velocity_override: Optional[int] = None) -> pd.DataFrame:
        """Applies the full transformation pipeline to the input dataframe."""
        # 1. Add country
        if 'country' not in df.columns and self.ip_map is not None:
            df['country'] = df['ip_address'].apply(self.get_country)
        
        # 2. Time features
        df = self.engineer_time_features(df)
        
        # 3. Velocity
        if velocity_override is not None:
            df['tx_count_last_24h'] = velocity_override
        else:
            df = self.calculate_velocity(df)
        
        # 4. Scaling
        num_cols = ['purchase_value', 'age', 'time_since_signup_hours', 'tx_count_last_24h']
        if is_training:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
            
        # 5. Encoding
        cat_cols = ['source', 'browser', 'sex', 'country', 'hour_of_day', 'day_of_week']
        if is_training:
            encoded_cats = self.encoder.fit_transform(df[cat_cols])
        else:
            encoded_cats = self.encoder.transform(df[cat_cols])
            
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=self.encoder.get_feature_names_out(cat_cols),
            index=df.index
        )
        
        # Drop raw columns and combine
        cols_to_drop = cat_cols + ['signup_time', 'purchase_time', 'ip_address', 'device_id', 'user_id']
        df_final = pd.concat([
            df.drop(columns=[c for c in cols_to_drop if c in df.columns]),
            encoded_df
        ], axis=1)
        
        return df_final

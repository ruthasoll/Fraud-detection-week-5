import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer
from src.features.feature_store import FeatureStore
from src.utils.config import Config
from datetime import datetime

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def fe(config):
    return FeatureEngineer(config)

@pytest.fixture
def fs():
    return FeatureStore(window_hours=24)

def test_engineer_time_features(fe):
    df = pd.DataFrame({
        'signup_time': [pd.Timestamp('2023-01-01 10:00:00')],
        'purchase_time': [pd.Timestamp('2023-01-01 12:00:00')]
    })
    df_res = fe.engineer_time_features(df)
    assert df_res['hour_of_day'].iloc[0] == 12
    assert df_res['day_of_week'].iloc[0] == 6  # 2023-01-01 was a Sunday (6)
    assert df_res['time_since_signup_hours'].iloc[0] == 2.0

def test_feature_store_velocity(fs):
    user_id = 123
    t1 = datetime(2023, 1, 1, 10, 0)
    t2 = datetime(2023, 1, 1, 11, 0)
    t3 = datetime(2023, 1, 3, 10, 0) # Outside 24h window
    
    v1 = fs.update_and_get_velocity(user_id, t1)
    assert v1 == 1
    
    v2 = fs.update_and_get_velocity(user_id, t2)
    assert v2 == 2
    
    v3 = fs.update_and_get_velocity(user_id, t3)
    assert v3 == 1 # t1 and t2 should be cleared

def test_get_country_unknown(fe):
    assert fe.get_country(123456) == "Unknown"

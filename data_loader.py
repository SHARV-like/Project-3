"""
Data loading and preprocessing module.
Uses a public dataset or generates realistic synthetic data if dataset unavailable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import config

def generate_synthetic_energy_data(n_samples=1000, random_state=42):
    """
    Generate synthetic energy consumption data based on weather patterns.
    This simulates a realistic public dataset for demonstration.
    
    Parameters:
    - n_samples: Number of samples to generate
    - random_state: Random seed for reproducibility
    
    Returns:
    - DataFrame with features and target (energy_consumption)
    """
    np.random.seed(random_state)
    
    # Generate date range
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Generate weather features (realistic ranges)
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + \
                  5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                  np.random.normal(0, 2, n_samples)
    
    humidity = 50 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + \
               np.random.normal(0, 5, n_samples)
    humidity = np.clip(humidity, 0, 100)
    
    pressure = 1013 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + \
               np.random.normal(0, 3, n_samples)
    
    wind_speed = 5 + 3 * np.abs(np.sin(2 * np.pi * np.arange(n_samples) / 24)) + \
                 np.random.exponential(2, n_samples)
    wind_speed = np.clip(wind_speed, 0, 30)
    
    cloud_coverage = 30 + 30 * np.abs(np.sin(2 * np.pi * np.arange(n_samples) / 24)) + \
                     np.random.normal(0, 10, n_samples)
    cloud_coverage = np.clip(cloud_coverage, 0, 100)
    
    # Extract temporal features
    hour = dates.hour
    day_of_week = dates.dayofweek
    month = dates.month
    
    # Generate energy consumption (target variable)
    # Energy consumption depends on:
    # - Temperature (heating/cooling)
    # - Hour of day (peak usage)
    # - Day of week (weekend vs weekday)
    base_consumption = 100
    
    # Temperature effect (U-shaped: high consumption at extremes)
    temp_effect = 0.5 * (temperature - 20) ** 2
    
    # Time of day effect (peak in morning and evening)
    hour_effect = 20 * np.sin(2 * np.pi * hour / 24) ** 2
    
    # Day of week effect (lower on weekends)
    day_effect = -10 if day_of_week < 5 else -20
    
    # Weather effects
    humidity_effect = 0.1 * humidity
    wind_effect = -0.5 * wind_speed  # Wind reduces heating needs
    
    energy_consumption = base_consumption + temp_effect + hour_effect + \
                        day_effect + humidity_effect + wind_effect + \
                        np.random.normal(0, 5, n_samples)
    energy_consumption = np.maximum(energy_consumption, 0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'datetime': dates,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'cloud_coverage': cloud_coverage,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'energy_consumption': energy_consumption
    })
    
    return data

def load_data(data_path=None):
    """
    Load energy consumption data.
    If data_path is provided and file exists, load from file.
    Otherwise, generate synthetic data.
    
    Parameters:
    - data_path: Path to CSV file (optional)
    
    Returns:
    - DataFrame with features and target
    """
    if data_path and Path(data_path).exists():
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        # Ensure datetime column is datetime type
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
    else:
        print("Generating synthetic energy consumption data...")
        data = generate_synthetic_energy_data(n_samples=2000, random_state=config.RANDOM_STATE)
        # Save for future use
        save_path = config.DATA_DIR / "energy_data.csv"
        data.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    return data

def prepare_features(data):
    """
    Prepare features for model training.
    
    Parameters:
    - data: DataFrame with raw features
    
    Returns:
    - X: Feature matrix
    - y: Target vector
    - feature_names: List of feature names
    """
    feature_cols = [
        'temperature',
        'humidity',
        'pressure',
        'wind_speed',
        'cloud_coverage',
        'hour',
        'day_of_week',
        'month'
    ]
    
    X = data[feature_cols].values
    y = data['energy_consumption'].values
    
    return X, y, feature_cols

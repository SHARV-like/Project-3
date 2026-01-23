"""
Risk mapper for Forest Fire Early Warning System.
Maps weather data to the feature format expected by the trained model.
"""

import pandas as pd
from typing import Dict


def map_weather_to_features(weather_data: Dict[str, float]) -> pd.DataFrame:
    """
    Map weather data dictionary to a single-row DataFrame with features
    in the exact order used during model training.
    
    Parameters:
    - weather_data: Dictionary with keys 'temperature', 'humidity', 'wind_speed', 'rainfall'
    
    Returns:
    - Single-row pandas DataFrame with columns in exact training order:
        temperature, humidity, wind, rain, FFMC, DMC, DC, ISI
        All values are float type
    """
    # Fixed default values for fire weather indices
    # These defaults represent moderate fire danger conditions
    # FFMC (Fine Fuel Moisture Code): 0-101, default 70 (moderate)
    # DMC (Duff Moisture Code): 0-300+, default 30 (moderate)
    # DC (Drought Code): 0-800+, default 100 (moderate)
    # ISI (Initial Spread Index): 0-56, default 5 (moderate)
    DEFAULT_FFMC = 70.0
    DEFAULT_DMC = 30.0
    DEFAULT_DC = 100.0
    DEFAULT_ISI = 5.0
    
    # Extract weather data values
    # Map from weather_api keys to feature names
    temperature = float(weather_data.get('temperature', 0.0))
    humidity = float(weather_data.get('humidity', 0.0))
    wind = float(weather_data.get('wind_speed', 0.0))  # Map from 'wind_speed' to 'wind'
    rain = float(weather_data.get('rainfall', 0.0))  # Map from 'rainfall' to 'rain'
    
    # Create DataFrame with EXACT feature order used during training
    # Order: temperature, humidity, wind, rain, FFMC, DMC, DC, ISI
    df = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'wind': [wind],
        'rain': [rain],
        'FFMC': [DEFAULT_FFMC],
        'DMC': [DEFAULT_DMC],
        'DC': [DEFAULT_DC],
        'ISI': [DEFAULT_ISI]
    })
    
    # Ensure all values are numeric (float)
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    return df

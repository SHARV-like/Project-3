"""
Predictor for Forest Fire Early Warning System.
Loads trained model and makes fire risk predictions.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def predict_fire_risk(feature_df: pd.DataFrame) -> Tuple[str, float]:
    """
    Predict forest fire risk using the trained model.
    
    Parameters:
    - feature_df: pandas DataFrame with features in the exact order used during training
                 Columns: temperature, humidity, wind, rain, FFMC, DMC, DC, ISI
                 Must be a single-row DataFrame
    
    Returns:
    - Tuple of (risk_label, confidence_score):
        risk_label: str - 'Low', 'Medium', or 'High'
        confidence_score: float - Probability of the predicted class (0.0 to 1.0)
    
    Raises:
    - FileNotFoundError: If model file is not found
    - ValueError: If feature_df has incorrect shape or columns
    """
    # Load trained model
    models_dir = Path(__file__).parent.parent / "models"
    model_path = models_dir / "fire_risk_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first using train_model.py"
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Validate input DataFrame
    if feature_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if feature_df.shape[0] != 1:
        raise ValueError(
            f"Input DataFrame must have exactly 1 row, but got {feature_df.shape[0]} rows"
        )
    
    # Convert DataFrame to numpy array for prediction
    # Ensure features are in the correct order
    # The model expects: temperature, humidity, wind_speed, rainfall, FFMC, DMC, DC, ISI
    # But risk_mapper uses: temperature, humidity, wind, rain, FFMC, DMC, DC, ISI
    # We need to map the column names correctly
    
    # Get expected feature order from model (if available) or use known order
    # Based on data_loader, the features are: temperature, humidity, wind_speed, rainfall, FFMC, DMC, DC, ISI
    # But risk_mapper uses: temperature, humidity, wind, rain, FFMC, DMC, DC, ISI
    # Map 'wind' -> 'wind_speed' and 'rain' -> 'rainfall' if needed
    
    # Create a mapping dictionary
    column_mapping = {
        'wind': 'wind_speed',
        'rain': 'rainfall'
    }
    
    # Create a copy of the DataFrame with potentially renamed columns
    df_copy = feature_df.copy()
    
    # Rename columns if they use short names
    for old_name, new_name in column_mapping.items():
        if old_name in df_copy.columns and new_name not in df_copy.columns:
            df_copy = df_copy.rename(columns={old_name: new_name})
    
    # Expected feature order (as used in training)
    expected_features = ['temperature', 'humidity', 'wind_speed', 'rainfall', 'FFMC', 'DMC', 'DC', 'ISI']
    
    # Check if all expected features are present
    missing_features = set(expected_features) - set(df_copy.columns)
    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}. "
            f"Expected features: {expected_features}"
        )
    
    # Select features in the exact order expected by the model
    X = df_copy[expected_features].values.astype(np.float64)
    
    # Get class probabilities
    probabilities = model.predict_proba(X)[0]  # Get first (and only) row
    
    # Get class labels
    class_labels = model.classes_
    
    # Find the class with highest probability
    max_prob_idx = np.argmax(probabilities)
    risk_label = str(class_labels[max_prob_idx])
    confidence_score = float(probabilities[max_prob_idx])
    
    # Ensure confidence_score is between 0 and 1
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    return risk_label, confidence_score

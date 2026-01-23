"""
Data loading and preprocessing module for Algerian Forest Fires dataset.
Loads, cleans, and prepares the dataset for fire risk prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def load_data():
    """
    Load Algerian Forest Fires dataset from data/raw/algerian_forest_fires.csv.
    Handles the two-region structure and cleans the data.
    
    Returns:
    - X: Feature matrix (numpy array)
    - y: Target labels (numpy array) - Low/Medium/High fire risk
    """
    # Path to dataset
    data_path = Path(__file__).parent.parent / "data" / "raw" / "algerian_forest_fires.csv"
    
    # Try alternative filename if primary doesn't exist
    if not data_path.exists():
        alt_path = Path(__file__).parent.parent / "data" / "raw" / "Algerian_forest_fires_dataset.csv"
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path} or {alt_path}")
    
    print(f"Loading dataset from {data_path}")
    
    # Read the CSV file
    # The dataset has two regions separated by empty rows and header rows
    df = pd.read_csv(data_path, skipinitialspace=True)
    
    # Remove empty rows and rows that are headers or region names
    df = df.dropna(subset=['day', 'month', 'year'], how='all')
    df = df[~df['day'].astype(str).str.contains('day', case=False, na=False)]
    df = df[~df['day'].astype(str).str.contains('Region', case=False, na=False)]
    
    # Clean column names - remove extra spaces
    df.columns = df.columns.str.strip()
    
    # Map original column names to cleaned names (after stripping)
    # Handle both 'Rain' and 'Rain ' variations
    rain_col = 'Rain' if 'Rain' in df.columns else 'Rain '
    
    # Convert all numeric columns
    numeric_cols = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', rain_col, 
                    'FFMC', 'DMC', 'DC', 'ISI']
    
    for col in numeric_cols:
        if col in df.columns:
            # Replace any non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean the 'Classes' column - remove extra spaces
    if 'Classes' in df.columns:
        df['Classes'] = df['Classes'].str.strip()
    
    # Select only required features: temperature, humidity, wind speed, rainfall, FFMC, DMC, DC, ISI
    feature_cols = ['Temperature', 'RH', 'Ws', rain_col, 'FFMC', 'DMC', 'DC', 'ISI']
    # Only use columns that actually exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Drop rows where all key features are missing
    df = df.dropna(subset=feature_cols, how='all')
    
    # Handle missing values - impute with median for numeric features
    for col in feature_cols:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0  # Fallback if all values are NaN
            df[col] = df[col].fillna(median_val)
    
    # Select features: temperature, humidity, wind speed, rainfall, and drought indices
    feature_cols_clean = {
        'Temperature': 'temperature',
        'RH': 'humidity',
        'Ws': 'wind_speed',
        rain_col: 'rainfall',
        'FFMC': 'FFMC',
        'DMC': 'DMC',
        'DC': 'DC',
        'ISI': 'ISI'
    }
    
    # Create feature dataframe with cleaned names
    X_df = pd.DataFrame()
    for old_col, new_col in feature_cols_clean.items():
        if old_col in df.columns:
            X_df[new_col] = df[old_col]
    
    # Convert to numpy array
    X = X_df.values.astype(np.float64)
    
    # Create fire_risk labels based on fire occurrence (Classes column)
    y_labels = []
    
    for idx, row in df.iterrows():
        class_val = str(row.get('Classes', '')).strip().lower() if pd.notna(row.get('Classes')) else ''
        
        # Check if fire occurred
        fire_occurred = 'fire' in class_val and 'not' not in class_val
        
        # Get fire indices for severity assessment
        ffmc_val = row.get('FFMC', 0) if pd.notna(row.get('FFMC')) else 0
        dc_val = row.get('DC', 0) if pd.notna(row.get('DC')) else 0
        isi_val = row.get('ISI', 0) if pd.notna(row.get('ISI')) else 0
        
        # Ensure numeric values
        try:
            ffmc_val = float(ffmc_val)
            dc_val = float(dc_val)
            isi_val = float(isi_val)
        except (ValueError, TypeError):
            ffmc_val = 0
            dc_val = 0
            isi_val = 0
        
        # Determine fire_risk level
        if fire_occurred:
            # Fire occurred - determine severity based on fire indices
            if ffmc_val >= 85 or dc_val >= 100 or isi_val >= 10:
                risk = 'High'
            else:
                risk = 'Medium'
        else:
            # No fire occurred - determine risk based on fire danger conditions
            if ffmc_val >= 85 or dc_val >= 100 or isi_val >= 10:
                risk = 'High'  # High fire danger conditions
            elif ffmc_val >= 70 or dc_val >= 50 or isi_val >= 5:
                risk = 'Medium'
            else:
                risk = 'Low'
        
        y_labels.append(risk)
    
    y = np.array(y_labels)
    
    # Print dataset information
    print(f"\nDataset loaded successfully!")
    print(f"Dataset shape: {X.shape}")
    print(f"\nfire_risk distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    return X, y

if __name__ == "__main__":
    # Test the data loader
    X, y = load_data()
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")

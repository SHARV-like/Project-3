"""
Model training script.
Trains RandomForest and GradientBoosting models with fixed random_state.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import config
from data_loader import load_data, prepare_features

def train_models():
    """Train both RandomForest and GradientBoosting models."""
    
    print("=" * 60)
    print("Training Energy Consumption Prediction Models")
    print("=" * 60)
    
    # Load data
    data = load_data()
    X, y, feature_names = prepare_features(data)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split data (with fixed random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    models = {}
    results = {}
    
    # Train RandomForest
    print("\n" + "-" * 60)
    print("Training RandomForest Regressor...")
    print("-" * 60)
    
    rf_model = RandomForestRegressor(**config.RANDOM_FOREST_PARAMS)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    rf_results = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    print(f"Train MAE: {rf_results['train_mae']:.2f}")
    print(f"Train RMSE: {rf_results['train_rmse']:.2f}")
    print(f"Train R²: {rf_results['train_r2']:.4f}")
    print(f"Test MAE: {rf_results['test_mae']:.2f}")
    print(f"Test RMSE: {rf_results['test_rmse']:.2f}")
    print(f"Test R²: {rf_results['test_r2']:.4f}")
    
    models['random_forest'] = rf_model
    results['random_forest'] = rf_results
    
    # Save RandomForest model
    rf_path = config.MODELS_DIR / "random_forest_model.joblib"
    joblib.dump(rf_model, rf_path)
    print(f"\nRandomForest model saved to {rf_path}")
    
    # Train GradientBoosting
    print("\n" + "-" * 60)
    print("Training Gradient Boosting Regressor...")
    print("-" * 60)
    
    gb_model = GradientBoostingRegressor(**config.GRADIENT_BOOSTING_PARAMS)
    gb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = gb_model.predict(X_train)
    y_pred_test = gb_model.predict(X_test)
    
    gb_results = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    print(f"Train MAE: {gb_results['train_mae']:.2f}")
    print(f"Train RMSE: {gb_results['train_rmse']:.2f}")
    print(f"Train R²: {gb_results['train_r2']:.4f}")
    print(f"Test MAE: {gb_results['test_mae']:.2f}")
    print(f"Test RMSE: {gb_results['test_rmse']:.2f}")
    print(f"Test R²: {gb_results['test_r2']:.4f}")
    
    models['gradient_boosting'] = gb_model
    results['gradient_boosting'] = gb_results
    
    # Save GradientBoosting model
    gb_path = config.MODELS_DIR / "gradient_boosting_model.joblib"
    joblib.dump(gb_model, gb_path)
    print(f"\nGradientBoosting model saved to {gb_path}")
    
    # Feature importance comparison
    print("\n" + "=" * 60)
    print("Feature Importance Comparison")
    print("=" * 60)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'random_forest': rf_model.feature_importances_,
        'gradient_boosting': gb_model.feature_importances_
    })
    importance_df = importance_df.sort_values('random_forest', ascending=False)
    
    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))
    
    # Save feature importance
    importance_path = config.RESULTS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to {importance_path}")
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_path = config.RESULTS_DIR / "model_results.csv"
    results_df.to_csv(results_path)
    print(f"Model results saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return models, results

if __name__ == "__main__":
    train_models()

"""
Model explanation script for Forest Fire Early Warning System.
Uses SHAP to explain model predictions and feature importance.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import shap

from data_loader import load_data

def explain_model():
    """
    Generate SHAP explanations for the forest fire risk prediction model.
    """
    print("=" * 70)
    print("Forest Fire Early Warning System - Model Explanation")
    print("=" * 70)
    
    # Load trained model
    print("\nLoading trained model...")
    models_dir = Path(__file__).parent.parent / "models"
    model_path = models_dir / "fire_risk_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"  Model loaded from: {model_path}")
    
    # Load dataset
    print("\nLoading dataset...")
    X, y = load_data()
    
    # Get feature names
    feature_names = [
        'temperature', 'humidity', 'wind_speed', 'rainfall',
        'FFMC', 'DMC', 'DC', 'ISI'
    ]
    
    # Split data to get test subset
    print("\nPreparing test subset for SHAP analysis...")
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Use a subset for SHAP computation (for efficiency)
    n_samples = min(100, X_test.shape[0])
    X_shap = X_test[:n_samples]
    y_shap = y_test[:n_samples]
    
    print(f"  Using {n_samples} samples for SHAP analysis")
    
    # Initialize SHAP TreeExplainer
    print("\nInitializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_shap)
    
    # Handle multi-class output (SHAP returns list for multi-class)
    if isinstance(shap_values, list):
        # For multi-class, use the values for all classes
        # We'll use the mean across classes for global importance
        shap_values_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values_mean = np.abs(shap_values)
    
    print("  SHAP values computed successfully!")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "outputs" / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving plots to: {output_dir}")
    
    # 1. Global SHAP summary plot
    print("\n1. Generating global SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    
    if isinstance(shap_values, list):
        # For multi-class, create summary plot for each class
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names, 
                         class_names=model.classes_, show=False)
    else:
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    
    plt.tight_layout()
    summary_path = output_dir / "shap_summary_plot.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {summary_path}")
    
    # 2. SHAP bar plot (mean absolute importance)
    print("\n2. Generating SHAP bar plot (mean absolute importance)...")
    plt.figure(figsize=(10, 6))
    
    if isinstance(shap_values, list):
        # Average absolute SHAP values across all classes and samples
        # shap_values is a list of arrays, each array is (n_samples, n_features)
        abs_shap_list = [np.abs(sv) for sv in shap_values]  # List of (n_samples, n_features)
        stacked = np.stack(abs_shap_list, axis=0)  # (n_classes, n_samples, n_features)
        mean_abs_shap = np.mean(stacked, axis=(0, 1))  # Average over classes and samples -> (n_features,)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Ensure it's a 1D array and convert to list for easier handling
    mean_abs_shap = np.array(mean_abs_shap).flatten()
    
    # Create bar plot
    feature_importance = {name: float(val) for name, val in zip(feature_names, mean_abs_shap)}
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    plt.barh(features, importances, color='steelblue')
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance (Mean Absolute SHAP Values)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    bar_path = output_dir / "shap_bar_plot.png"
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {bar_path}")
    
    # 3. Local explanation for one high-risk sample
    print("\n3. Generating local explanation for high-risk sample...")
    
    # Find a high-risk sample
    high_risk_indices = np.where(y_shap == 'High')[0]
    if len(high_risk_indices) > 0:
        sample_idx = high_risk_indices[0]
    else:
        # If no high-risk in subset, use first sample
        sample_idx = 0
    
    sample_X = X_shap[sample_idx:sample_idx+1]
    sample_y = y_shap[sample_idx]
    sample_pred = model.predict(sample_X)[0]
    sample_proba = model.predict_proba(sample_X)[0]
    
    print(f"  Sample index: {sample_idx}")
    print(f"  True label: {sample_y}")
    print(f"  Predicted: {sample_pred}")
    print(f"  Probabilities: {dict(zip(model.classes_, sample_proba))}")
    
    # Get SHAP values for this sample
    if isinstance(shap_values, list):
        # For multi-class, use SHAP values for the predicted class
        pred_class_idx = list(model.classes_).index(sample_pred)
        sample_shap = shap_values[pred_class_idx][sample_idx]
        expected_value = explainer.expected_value[pred_class_idx]
    else:
        sample_shap = shap_values[sample_idx]
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = float(expected_value[0])
    
    # Ensure sample_shap is a 1D array and convert to list of floats
    sample_shap = np.array(sample_shap).flatten()
    sample_shap = [float(x) for x in sample_shap]
    
    # Create bar plot for local explanation
    plt.figure(figsize=(10, 8))
    
    # Sort features by absolute SHAP value
    contributions = {name: float(val) for name, val in zip(feature_names, sample_shap)}
    sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    features = [f[0] for f in sorted_contrib]
    values = [f[1] for f in sorted_contrib]
    colors = ['red' if v > 0 else 'blue' for v in values]
    
    plt.barh(features, values, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('SHAP Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Local Explanation: {sample_pred} Risk Prediction\n(True: {sample_y}, Base Value: {expected_value:.4f})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    local_path = output_dir / "shap_local_explanation.png"
    plt.savefig(local_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {local_path}")
    
    # Print feature contributions for the sample
    print(f"\n  Feature contributions for sample:")
    contributions = dict(zip(feature_names, sample_shap))
    sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, contrib in sorted_contrib:
        print(f"    {feat:15s}: {contrib:8.4f}")
    
    print("\n" + "=" * 70)
    print("Model explanation completed successfully!")
    print(f"All plots saved to: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    explain_model()

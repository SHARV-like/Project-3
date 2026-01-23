"""
Explainability module using SHAP values.
Provides model interpretability for predictions.
"""

import numpy as np
import shap
import joblib
import config
from pathlib import Path

class ModelExplainer:
    """Wrapper for SHAP-based model explanation."""
    
    def __init__(self, model, feature_names, model_type='random_forest'):
        """
        Initialize explainer.
        
        Parameters:
        - model: Trained model
        - feature_names: List of feature names
        - model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        
        # Initialize SHAP explainer based on model type
        if model_type == 'random_forest':
            # TreeExplainer is fast and exact for tree models
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.TreeExplainer(model)
    
    def explain_prediction(self, X, sample_idx=0):
        """
        Explain a single prediction.
        
        Parameters:
        - X: Feature matrix
        - sample_idx: Index of sample to explain
        
        Returns:
        - Dictionary with explanation details
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X[sample_idx:sample_idx+1])
        
        # Get prediction
        prediction = self.model.predict(X[sample_idx:sample_idx+1])[0]
        
        # Get base value (expected value)
        # For regression, expected_value is a scalar
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])
        else:
            base_value = float(base_value)
        
        # Create explanation dictionary
        explanation = {
            'prediction': float(prediction),
            'base_value': base_value,
            'feature_contributions': {}
        }
        
        # Map SHAP values to feature names
        # Handle both 1D and 2D shap_values arrays
        shap_vals = shap_values[0] if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1 else shap_values
        for i, feature_name in enumerate(self.feature_names):
            contribution = float(shap_vals[i])
            explanation['feature_contributions'][feature_name] = contribution
        
        # Sort by absolute contribution
        explanation['sorted_contributions'] = sorted(
            explanation['feature_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return explanation
    
    def explain_batch(self, X, max_samples=100):
        """
        Explain a batch of predictions (for summary statistics).
        
        Parameters:
        - X: Feature matrix
        - max_samples: Maximum number of samples to explain
        
        Returns:
        - Dictionary with summary statistics
        """
        # Limit samples for performance
        n_samples = min(X.shape[0], max_samples)
        X_sample = X[:n_samples]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        # Calculate mean absolute SHAP values (feature importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create summary
        summary = {
            'n_samples': n_samples,
            'mean_abs_shap': {
                feature_name: float(mean_abs_shap[i])
                for i, feature_name in enumerate(self.feature_names)
            }
        }
        
        # Sort by importance
        summary['sorted_importance'] = sorted(
            summary['mean_abs_shap'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return summary
    
    def get_feature_importance_ranking(self, X_sample=None, n_samples=100):
        """
        Get feature importance ranking based on SHAP values.
        
        Parameters:
        - X_sample: Sample data (optional, uses training data if not provided)
        - n_samples: Number of samples to use
        
        Returns:
        - List of (feature_name, importance) tuples, sorted by importance
        """
        if X_sample is None:
            # This would require access to training data
            # For now, use model's built-in feature importance
            importances = self.model.feature_importances_
            ranking = list(zip(self.feature_names, importances))
            ranking.sort(key=lambda x: x[1], reverse=True)
            return ranking
        
        summary = self.explain_batch(X_sample, max_samples=n_samples)
        return summary['sorted_importance']

def load_explainer(model_path, model_type='random_forest'):
    """
    Load a trained model and create an explainer.
    
    Parameters:
    - model_path: Path to saved model
    - model_type: Type of model
    
    Returns:
    - ModelExplainer instance
    """
    model = joblib.load(model_path)
    explainer = ModelExplainer(model, config.FEATURE_NAMES, model_type)
    return explainer

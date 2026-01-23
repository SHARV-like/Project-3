"""
Model training script for Forest Fire Early Warning System.
Trains a RandomForestClassifier for fire risk prediction.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from data_loader import load_data

def train_model():
    """
    Train RandomForestClassifier for forest fire risk prediction.
    """
    print("=" * 70)
    print("Forest Fire Early Warning System - Model Training")
    print("=" * 70)
    
    # Load data
    print("\nLoading dataset...")
    X, y = load_data()
    
    print(f"\nDataset loaded:")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    # Split data into train/test (stratified)
    print("\nSplitting data into train/test sets (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Train RandomForestClassifier
    print("\nTraining RandomForestClassifier...")
    print("  Parameters:")
    print("    n_estimators=300")
    print("    max_depth=None")
    print("    class_weight='balanced'")
    print("    random_state=42")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=1
    )
    
    model.fit(X_train, y_train)
    print("  Training completed!")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print(f"  Probability predictions shape: {y_pred_proba.shape}")
    
    # Evaluate model
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision (macro-averaged)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Precision (macro): {precision:.4f} ({precision*100:.2f}%)")
    
    # Recall (macro-averaged) - emphasized
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n*** RECALL (macro): {recall:.4f} ({recall*100:.2f}%) ***")
    print("   (Emphasized metric for early warning system)")
    
    # Per-class metrics
    print("\nPer-class metrics:")
    class_report = classification_report(y_test, y_pred, zero_division=0)
    print(class_report)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    classes = model.classes_
    
    # Print header
    print(" " * 12, end="")
    for cls in classes:
        print(f"{cls:>10}", end="")
    print()
    
    # Print matrix
    for i, cls in enumerate(classes):
        print(f"{cls:>10}  ", end="")
        for j in range(len(classes)):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    # Print row totals
    print(" " * 12, end="")
    for j in range(len(classes)):
        print(f"{cm[:, j].sum():>10}", end="")
    print(" (Predicted)")
    
    # Save model
    print("\n" + "=" * 70)
    print("Saving model...")
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "fire_risk_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  Model saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    
    return model, X_test, y_test, y_pred, y_pred_proba

if __name__ == "__main__":
    train_model()

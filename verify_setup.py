"""
Quick verification script to test if the project is set up correctly.
"""

import sys

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking imports...")
    try:
        import numpy
        print("✓ numpy")
        import pandas
        print("✓ pandas")
        import sklearn
        print("✓ scikit-learn")
        import shap
        print("✓ shap")
        import requests
        print("✓ requests")
        import joblib
        print("✓ joblib")
        print("\nAll imports successful!")
        return True
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_config():
    """Check if configuration is set up correctly."""
    print("\nChecking configuration...")
    try:
        import config
        print(f"✓ Random state: {config.RANDOM_STATE}")
        print(f"✓ Data directory: {config.DATA_DIR}")
        print(f"✓ Models directory: {config.MODELS_DIR}")
        print(f"✓ Results directory: {config.RESULTS_DIR}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def check_modules():
    """Check if project modules can be imported."""
    print("\nChecking project modules...")
    try:
        from data_loader import load_data
        print("✓ data_loader")
        from weather_api import WeatherAPI
        print("✓ weather_api")
        from explainability import ModelExplainer
        print("✓ explainability")
        print("\nAll modules imported successfully!")
        return True
    except Exception as e:
        print(f"✗ Module import error: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Project Setup Verification")
    print("=" * 60)
    
    all_ok = True
    all_ok &= check_imports()
    all_ok &= check_config()
    all_ok &= check_modules()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ Setup verification PASSED!")
        print("\nNext steps:")
        print("1. Run 'python train_model.py' to train models")
        print("2. Run 'python demo.py' to see the demo")
    else:
        print("✗ Setup verification FAILED!")
        print("Please fix the errors above.")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

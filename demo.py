"""
Main demo script for Weather-Based Energy Consumption Prediction.
Demonstrates the complete pipeline with real-time weather data.
"""

import numpy as np
import joblib
import config
from weather_api import WeatherAPI
from explainability import ModelExplainer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def format_prediction(prediction):
    """Format prediction for display."""
    return f"{prediction:.2f} kWh"

def format_explanation(explanation):
    """Format explanation for display."""
    lines = []
    lines.append(f"\nPredicted Energy Consumption: {format_prediction(explanation['prediction'])}")
    lines.append(f"Base Value (Average): {format_prediction(explanation['base_value'])}")
    lines.append("\nFeature Contributions:")
    lines.append("-" * 60)
    
    for feature, contribution in explanation['sorted_contributions']:
        sign = "+" if contribution >= 0 else ""
        lines.append(f"  {feature:20s}: {sign}{contribution:7.2f} kWh")
    
    return "\n".join(lines)

def prepare_features_from_weather(weather_data):
    """
    Prepare feature vector from weather data.
    
    Parameters:
    - weather_data: Dictionary with weather features
    
    Returns:
    - numpy array of features
    """
    features = np.array([
        weather_data['temperature'],
        weather_data['humidity'],
        weather_data['pressure'],
        weather_data['wind_speed'],
        weather_data['cloud_coverage'],
        weather_data['hour'],
        weather_data['day_of_week'],
        weather_data['month']
    ])
    
    return features.reshape(1, -1)

def run_demo(city="London", country_code="GB", model_type="random_forest"):
    """
    Run the complete demo pipeline.
    
    Parameters:
    - city: City name for weather data
    - country_code: Country code
    - model_type: 'random_forest' or 'gradient_boosting'
    """
    print("=" * 70)
    print("Weather-Based Energy Consumption Prediction System")
    print("=" * 70)
    
    # Check if model exists
    model_path = config.MODELS_DIR / f"{model_type}_model.joblib"
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please run 'python train_model.py' first to train the models.")
        return
    
    # Load model
    print(f"\nLoading {model_type} model...")
    model = joblib.load(model_path)
    explainer = ModelExplainer(model, config.FEATURE_NAMES, model_type)
    print("Model loaded successfully!")
    
    # Get weather data
    print(f"\nFetching weather data for {city}, {country_code}...")
    weather_api = WeatherAPI()
    weather_data = weather_api.get_weather_features(city, country_code)
    
    if weather_data.get('fallback'):
        print(f"⚠️  Using fallback weather data (API unavailable)")
    else:
        print("✓ Weather data fetched successfully")
    
    # Display weather information
    print("\n" + "-" * 70)
    print("Current Weather Conditions:")
    print("-" * 70)
    print(f"  City: {weather_data['city']}")
    print(f"  Temperature: {weather_data['temperature']:.1f}°C")
    print(f"  Humidity: {weather_data['humidity']:.1f}%")
    print(f"  Pressure: {weather_data['pressure']:.1f} hPa")
    print(f"  Wind Speed: {weather_data['wind_speed']:.1f} m/s")
    print(f"  Cloud Coverage: {weather_data['cloud_coverage']:.1f}%")
    print(f"  Hour: {weather_data['hour']}")
    print(f"  Day of Week: {weather_data['day_of_week']} ({'Mon' if weather_data['day_of_week'] < 5 else 'Weekend'})")
    print(f"  Month: {weather_data['month']}")
    
    # Prepare features
    X = prepare_features_from_weather(weather_data)
    
    # Make prediction
    print("\n" + "-" * 70)
    print("Making Prediction...")
    print("-" * 70)
    
    prediction = model.predict(X)[0]
    print(f"\nPredicted Energy Consumption: {format_prediction(prediction)}")
    
    # Get explanation
    print("\n" + "-" * 70)
    print("Model Explanation (SHAP Values):")
    print("-" * 70)
    
    explanation = explainer.explain_prediction(X, sample_idx=0)
    print(format_explanation(explanation))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Based on current weather conditions in {city}, the model predicts")
    print(f"an energy consumption of {format_prediction(prediction)}.")
    print("\nThe top contributing factors are:")
    for i, (feature, contribution) in enumerate(explanation['sorted_contributions'][:3], 1):
        print(f"  {i}. {feature}: {contribution:+.2f} kWh")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)

def interactive_demo():
    """Run interactive demo with user input."""
    print("\n" + "=" * 70)
    print("Interactive Demo")
    print("=" * 70)
    
    # Get city input
    city = input("\nEnter city name (default: London): ").strip() or "London"
    country_code = input("Enter country code (default: GB): ").strip() or "GB"
    
    # Get model choice
    print("\nAvailable models:")
    print("  1. Random Forest (default)")
    print("  2. Gradient Boosting")
    model_choice = input("Select model (1 or 2, default: 1): ").strip() or "1"
    model_type = "random_forest" if model_choice == "1" else "gradient_boosting"
    
    run_demo(city, country_code, model_type)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command-line mode
        city = sys.argv[1] if len(sys.argv) > 1 else "London"
        country_code = sys.argv[2] if len(sys.argv) > 2 else "GB"
        model_type = sys.argv[3] if len(sys.argv) > 3 else "random_forest"
        run_demo(city, country_code, model_type)
    else:
        # Interactive mode
        interactive_demo()

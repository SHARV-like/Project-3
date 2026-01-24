"""
Backend verification script for Forest Fire Early Warning System.
Tests the complete pipeline: weather fetch -> feature extraction -> prediction.
"""

import sys

# Fixed coordinates for verification (Algerian Forest region)
LATITUDE = 36.6500
LONGITUDE = 3.1167


def verify_backend_pipeline():
    """
    Verify the backend pipeline by:
    1. Fetching weather data using fixed coordinates
    2. Building feature vector from weather data
    3. Predicting fire risk using the trained model
    4. Displaying all outputs clearly
    """
    print("=" * 70)
    print("   FOREST FIRE EARLY WARNING SYSTEM - BACKEND VERIFICATION")
    print("=" * 70)
    print()
    
    # Display fixed coordinates being used
    print(f"Using fixed coordinates:")
    print(f"  Latitude:  {LATITUDE}")
    print(f"  Longitude: {LONGITUDE}")
    print()
    
    # Step 1: Fetch weather data
    print("-" * 70)
    print("STEP 1: Fetching Weather Data")
    print("-" * 70)
    
    try:
        from src.weather_api import get_weather_data
        weather_data = get_weather_data(LATITUDE, LONGITUDE)
        
        print("Weather Data Dictionary:")
        print(f"  Temperature: {weather_data['temperature']:.2f} °C")
        print(f"  Humidity:    {weather_data['humidity']:.1f} %")
        print(f"  Wind Speed:  {weather_data['wind_speed']:.2f} m/s")
        print(f"  Rainfall:    {weather_data['rainfall']:.2f} mm")
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to fetch weather data: {e}")
        print()
        return False
    
    # Step 2: Build feature vector
    print("-" * 70)
    print("STEP 2: Building Feature Vector")
    print("-" * 70)
    
    try:
        from src.risk_mapper import map_weather_to_features
        feature_df = map_weather_to_features(weather_data)
        
        print("Feature DataFrame:")
        print(feature_df.to_string(index=False))
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to build feature vector: {e}")
        print()
        return False
    
    # Step 3: Predict fire risk
    print("-" * 70)
    print("STEP 3: Predicting Fire Risk")
    print("-" * 70)
    
    try:
        from src.predictor import predict_fire_risk
        risk_label, confidence_score = predict_fire_risk(feature_df)
        
        print(f"Predicted Fire Risk Label: {risk_label}")
        print(f"Confidence Score:          {confidence_score:.4f} ({confidence_score * 100:.2f}%)")
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to predict fire risk: {e}")
        print()
        return False
    
    # Success footer
    print("=" * 70)
    print("   BACKEND VERIFICATION COMPLETE - ALL STEPS PASSED")
    print("=" * 70)
    
    return True


def main():
    """Main entry point for the verification script."""
    success = verify_backend_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

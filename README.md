# Weather-Based Energy Consumption Prediction System

A complete, end-to-end machine learning project that predicts energy consumption based on weather conditions and temporal features. Built with explainability, determinism, and demo-readiness in mind.

## Features

- **Deterministic**: Fixed random states ensure reproducible results
- **Explainable**: SHAP values provide model interpretability
- **Real-time Weather**: Integrates with OpenWeather API
- **Graceful Degradation**: Falls back to default values if API is unavailable
- **No Deep Learning**: Uses RandomForest and GradientBoosting only
- **Demo-Ready**: Complete pipeline from data to prediction

## Project Structure

```
.
├── config.py              # Configuration and fixed random states
├── data_loader.py         # Data loading and preprocessing
├── weather_api.py         # OpenWeather API integration
├── train_model.py         # Model training script
├── explainability.py      # SHAP-based model explanation
├── demo.py                # Main demo script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
├── data/                 # Data directory (auto-created)
├── models/               # Trained models (auto-created)
└── results/              # Results and outputs (auto-created)
```

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenWeather API (optional but recommended):**
   - Sign up for a free API key at [OpenWeatherMap](https://openweathermap.org/api)
   - Create a `.env` file in the project root:
     ```
     OPENWEATHER_API_KEY=your_api_key_here
     ```
   - If no API key is provided, the system will use fallback weather data

## Usage

### 1. Train the Models

First, train the models on the dataset:

```bash
python train_model.py
```

This will:
- Generate/load the energy consumption dataset
- Train both RandomForest and GradientBoosting models
- Save models to `models/` directory
- Generate feature importance and evaluation metrics

### 2. Run the Demo

**Interactive mode:**
```bash
python demo.py
```

**Command-line mode:**
```bash
python demo.py London GB random_forest
```

Arguments:
- City name (default: London)
- Country code (default: GB)
- Model type: `random_forest` or `gradient_boosting` (default: random_forest)

### Example Output

```
======================================================================
Weather-Based Energy Consumption Prediction System
======================================================================

Loading random_forest model...
Model loaded successfully!

Fetching weather data for London, GB...
✓ Weather data fetched successfully

----------------------------------------------------------------------
Current Weather Conditions:
----------------------------------------------------------------------
  City: London
  Temperature: 12.5°C
  Humidity: 65.0%
  Pressure: 1015.0 hPa
  Wind Speed: 4.2 m/s
  Cloud Coverage: 75.0%
  Hour: 14
  Day of Week: 2 (Mon)
  Month: 11

----------------------------------------------------------------------
Making Prediction...
----------------------------------------------------------------------

Predicted Energy Consumption: 125.34 kWh

----------------------------------------------------------------------
Model Explanation (SHAP Values):
----------------------------------------------------------------------

Predicted Energy Consumption: 125.34 kWh
Base Value (Average): 115.50 kWh

Feature Contributions:
------------------------------------------------------------
  temperature          :  +5.23 kWh
  hour                 :  +3.45 kWh
  humidity             :  +1.20 kWh
  ...
```

## Model Details

### Algorithms Used
- **RandomForest Regressor**: Ensemble of decision trees
- **GradientBoosting Regressor**: Sequential ensemble with boosting

### Features
- Temperature (°C)
- Humidity (%)
- Pressure (hPa)
- Wind Speed (m/s)
- Cloud Coverage (%)
- Hour (0-23)
- Day of Week (0-6)
- Month (1-12)

### Determinism
- All random states fixed to `42`
- Same input always produces same output
- Reproducible across runs

## Explainability

The system uses SHAP (SHapley Additive exPlanations) values to explain predictions:
- Shows contribution of each feature to the prediction
- Highlights most important factors
- Provides interpretable model insights

## API Integration

### OpenWeather API
- Fetches real-time weather data
- Handles errors gracefully
- Falls back to default values if unavailable
- Rate-limited to avoid throttling

### Without API Key
The system works without an API key by using fallback weather values. This ensures the demo always runs, even if the API is unavailable.

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Determinism Guarantee

This project is designed to be fully deterministic:
- Fixed random seeds (`random_state=42`)
- No non-deterministic operations
- Same input → Same output
- Reproducible results across runs

## License

This project is provided as-is for educational and demonstration purposes.

## Notes

- The dataset is generated synthetically but follows realistic patterns
- Models are saved after training for reuse
- All outputs are saved to `results/` directory
- The system is designed for hackathon/demo purposes

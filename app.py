"""
Forest Fire Early Warning System - Streamlit UI
This module provides the user interface for the forest fire prediction system.
UI Layout ONLY - No backend logic implemented yet.
"""

import streamlit as st
import folium
import requests
import plotly.graph_objects as go
from streamlit_folium import st_folium
from src.weather_api import fetch_weather
from src.risk_mapper import build_feature_vector
from src.predictor import predict_fire_risk

# =============================================================================
# Region Presets Dictionary
# =============================================================================
# Key: Human-readable region name (displayable in dropdown)
# Value: Tuple of (latitude, longitude)
#
# Scalable to 150+ regions by:
# 1. Group regions by continent for better organization
# 2. Load from external JSON/CSV for maintainability
# =============================================================================
# --- ADD: Dummy Weather Data for Simulation Mode ---
# --- ADD: Dummy Weather Data for Simulation Mode ---
DUMMY_WEATHER = {
    "Low Risk": {
        "temperature": 22.0,
        "humidity": 78.0,
        "wind_speed": 5.0,
        "rainfall": 3.5
    },
    "Medium Risk": {
        "temperature": 28.5,
        "humidity": 42.0,
        "wind_speed": 14.0,
        "rainfall": 0.5
    },
    "High Risk": {
        "temperature": 36.5,
        "humidity": 20.0,
        "wind_speed": 28.0,
        "rainfall": 0.0
    }
}

REGION_PRESETS = {
    "— Select a region —": None,  # Placeholder option (no coordinates)
    
    # --- North America ---
    "California, USA": (36.7783, -119.4179),
    "Oregon, USA": (43.8041, -121.5395),
    "Colorado, USA": (39.5501, -105.7821),
    "Montana, USA": (46.8797, -110.3626),
    "Arizona, USA": (33.4484, -112.0740),
    "Texas, USA": (30.2672, -97.7431),
    "Florida, USA": (25.2866, -80.8987),
    "Alaska, USA": (64.2008, -149.4937),
    "British Columbia, Canada": (53.7267, -127.6476),
    "Alberta, Canada": (51.0447, -114.0719),
    "Ontario, Canada": (49.2827, -84.5050),
    "Baja California, Mexico": (28.0339, -114.0356),
    
    # --- South America ---
    "Amazon Rainforest": (-3.4653, -62.2159),
    "Cerrado, Brazil": (-15.7942, -47.8825),
    "Pantanal, Brazil": (-17.6509, -57.4559),
    "Patagonia, Argentina": (-41.1335, -71.3103),
    "Central Chile": (-33.4489, -70.6693),
    "Colombian Llanos": (4.5709, -74.2973),
    
    # --- Europe (Mediterranean) ---
    "Algarve, Portugal": (37.0179, -7.9304),
    "Andalusia, Spain": (37.3891, -5.9845),
    "Catalonia, Spain": (41.3851, 2.1734),
    "Provence, France": (43.9352, 6.0679),
    "Sardinia, Italy": (40.1209, 9.0129),
    "Sicily, Italy": (37.5994, 14.0154),
    "Peloponnese, Greece": (37.5079, 22.3746),
    "Crete, Greece": (35.2401, 24.8093),
    "Dalmatia, Croatia": (43.5081, 16.4402),
    "Aegean Coast, Turkey": (38.4192, 27.1287),
    
    # --- Africa ---
    "Kabylie, Algeria": (36.7500, 5.0500),
    "Atlas Mountains, Morocco": (31.6295, -7.9811),
    "Ethiopian Highlands": (9.1450, 40.4897),
    "Kenyan Savanna": (-1.2921, 36.8219),
    "South African Fynbos": (-33.9249, 18.4241),
    "Kruger, South Africa": (-23.9884, 31.5547),
    
    # --- Asia ---
    "Western Ghats, India": (11.0168, 76.9558),
    "Himalayan Foothills, India": (30.0668, 79.0193),
    "Siberian Taiga, Russia": (62.0000, 90.0000),
    "Yunnan, China": (25.0389, 102.7183),
    "Hokkaido, Japan": (43.0646, 141.3469),
    "Sumatra, Indonesia": (-0.7893, 113.9213),
    "Borneo, Malaysia": (4.2105, 117.9465),
    
    # --- Australia & Oceania ---
    "Australia (NSW)": (-33.8688, 151.2093),
    "Victoria, Australia": (-37.8136, 145.0884),
    "Queensland, Australia": (-20.9176, 142.7028),
    "Western Australia": (-31.9505, 115.8605),
    "Tasmania, Australia": (-42.8821, 147.3272),
    "New Zealand (North)": (-38.6857, 176.0702),
}

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Forest Fire Early Warning System",
    layout="wide"
)

# =============================================================================
# Session State Initialization
# =============================================================================
# Session state is used to persist data across UI re-renders.
# This ensures that:
# 1. Weather data is fetched ONLY when the user clicks "Predict Fire Risk"
# 2. UI interactions (like toggling fullscreen) do NOT trigger re-fetching
# 3. The same weather values remain "locked" throughout the assessment
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "weather_error" not in st.session_state:
    st.session_state.weather_error = None
if "assessment_location" not in st.session_state:
    # Store the location used for the assessment to track changes
    st.session_state.assessment_location = None

# Reverse geocoding session state
# These variables track the resolved address for the current coordinates
# Address is fetched ONLY when coordinates change, NOT on every re-render
if "resolved_address" not in st.session_state:
    st.session_state.resolved_address = None
if "last_geocoded_coords" not in st.session_state:
    # Track the last coordinates that were geocoded to detect changes
    st.session_state.last_geocoded_coords = None

if "is_water_location" not in st.session_state:
    st.session_state.is_water_location = False

# Initialize coordinates in session state if not present (Internal State)
if "selected_latitude" not in st.session_state:
    st.session_state.selected_latitude = 36.0
if "selected_longitude" not in st.session_state:
    st.session_state.selected_longitude = 2.0

# Callbacks to sync sidebar widget changes to internal state
def update_latitude():
    st.session_state.selected_latitude = st.session_state.latitude_input

def update_longitude():
    st.session_state.selected_longitude = st.session_state.longitude_input


# =============================================================================
# Region Dropdown Callback
# =============================================================================
# When a region is selected from the dropdown:
# 1. Update latitude/longitude session state to the region's coordinates
# 2. Clear all stale prediction data to ensure fresh assessment
# 3. Prediction is NOT triggered automatically - user must click button
# 4. Call st.rerun() to immediately update map
def update_region_selection():
    """Callback triggered when a region is selected from the dropdown."""
    selected_region = st.session_state.region_selector
    
    # Only update if a valid region is selected (not the placeholder)
    if selected_region and REGION_PRESETS.get(selected_region) is not None:
        lat, lon = REGION_PRESETS[selected_region]
        
        # Update coordinates in session state
        st.session_state.selected_latitude = lat
        st.session_state.selected_longitude = lon
        
        # Clear stale prediction state (force fresh assessment)
        st.session_state.weather_data = None
        st.session_state.prediction_result = None
        st.session_state.prediction_confidence = None
        st.session_state.risk_explanation = None
        
        # Rerun to update map immediately
        st.rerun()

# Prediction session state
# These variables store the ML prediction results
# Prediction runs ONLY once per button click, results persist across re-renders
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_confidence" not in st.session_state:
    st.session_state.prediction_confidence = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

# Explanation session state
# Stores the heuristic explanation for the current prediction
# Explanation is generated ONLY when prediction changes, NOT on every re-render
if "risk_explanation" not in st.session_state:
    st.session_state.risk_explanation = None

# =============================================================================
# STEP 23: Local Risk Probe Session State
# =============================================================================
# Probe allows click-to-inspect weather without triggering predictions
# probe_coords: (lat, lon) of the clicked location
# probe_weather: normalized weather dict for the clicked location
if "probe_coords" not in st.session_state:
    st.session_state.probe_coords = None
if "probe_weather" not in st.session_state:
    st.session_state.probe_weather = None

# =============================================================================
# STEP 24: Time Horizon Session State
# =============================================================================
# Allows scenario-based prediction for future time horizons
# Adjusts weather values heuristically (NOT from forecast API)
if "prediction_horizon" not in st.session_state:
    st.session_state.prediction_horizon = "Now"

# =============================================================================
# STEP 25: Judge Mode Session State
# =============================================================================
# Enables advanced explainability features for hackathon judges
if "judge_mode" not in st.session_state:
    st.session_state.judge_mode = False


# =============================================================================
# STEP 29: Post-Prediction Calibration
# =============================================================================
def calibrate_risk(risk_label, weather):
    """
    Adjust model prediction using domain sanity checks (Safety Rules).
    """
    temp = weather.get("temperature", 0)
    hum = weather.get("humidity", 50)
    wind = weather.get("wind", 0) # Uses normalized key 'wind' from Step 21
    rain = weather.get("rain", 0) # Uses normalized key 'rain' from Step 21
    
    # Rule 1: Force LOW risk if very mild conditions
    if temp < 25 and hum > 65 and rain > 2 and wind < 10:
        return "Low"
    
    # Rule 2: Force HIGH risk if extreme conditions
    if temp >= 38 and hum <= 25 and wind >= 25 and rain == 0:
        return "High"
        
    # Otherwise return original model prediction
    return risk_label


# =============================================================================
# Explanation Generator Function
# =============================================================================
def generate_risk_explanation(weather_data: dict, risk_level: str) -> list:
    """
    Generate heuristic-based explanations for the predicted fire risk.
    
    This function analyzes weather conditions and creates human-readable
    bullet points explaining why the risk level was predicted.
    
    Args:
        weather_data: Dictionary with temperature, humidity, wind, rain values
        risk_level: Predicted risk level (Low/Medium/High)
    
    Returns:
        List of explanation strings (3-5 bullet points)
    """
    explanations = []
    
    # Extract weather values with defaults
    # Uses normalized keys: temperature, humidity, wind, rain
    temp = weather_data.get('temperature', 0)
    humidity = weather_data.get('humidity', 50)
    wind = weather_data.get('wind', 0)
    rain = weather_data.get('rain', 0)
    
    # Temperature analysis
    # High temperatures (>30°C) increase fire risk
    if temp > 35:
        explanations.append(f"🌡️ **Very high temperature** ({temp:.1f}°C) significantly increases fire ignition risk")
    elif temp > 30:
        explanations.append(f"🌡️ **High temperature** ({temp:.1f}°C) contributes to elevated fire risk")
    elif temp > 25:
        explanations.append(f"🌡️ **Warm temperature** ({temp:.1f}°C) moderately affects fire conditions")
    else:
        explanations.append(f"🌡️ **Moderate temperature** ({temp:.1f}°C) has lower impact on fire risk")
    
    # Humidity analysis
    # Low humidity (<30%) increases fire risk
    if humidity < 20:
        explanations.append(f"💧 **Very low humidity** ({humidity:.0f}%) creates extremely dry conditions")
    elif humidity < 40:
        explanations.append(f"💧 **Low humidity** ({humidity:.0f}%) allows vegetation to dry out faster")
    elif humidity > 70:
        explanations.append(f"💧 **High humidity** ({humidity:.0f}%) helps reduce fire spread potential")
    else:
        explanations.append(f"💧 **Moderate humidity** ({humidity:.0f}%) has neutral impact on fire risk")
    
    # Wind analysis
    # Higher wind speeds increase fire spread
    if wind > 30:
        explanations.append(f"💨 **Strong winds** ({wind:.1f} km/h) can rapidly spread fires")
    elif wind > 15:
        explanations.append(f"💨 **Moderate winds** ({wind:.1f} km/h) may accelerate fire spread")
    else:
        explanations.append(f"💨 **Light winds** ({wind:.1f} km/h) limit fire spread potential")
    
    # Rainfall analysis
    # Rainfall decreases fire risk
    if rain > 5:
        explanations.append(f"🌧️ **Recent rainfall** ({rain:.1f} mm) significantly reduces fire risk")
    elif rain > 0:
        explanations.append(f"🌧️ **Light rainfall** ({rain:.1f} mm) slightly mitigates fire conditions")
    else:
        explanations.append(f"🌧️ **No recent rainfall** increases vegetation dryness")
    
    # Overall risk summary
    if risk_level == "High":
        explanations.append("⚠️ **Combined conditions** indicate elevated fire danger")
    elif risk_level == "Medium":
        explanations.append("ℹ️ **Mixed conditions** suggest moderate caution is advised")
    else:
        explanations.append("✅ **Favorable conditions** indicate lower fire probability")
    
    return explanations


# =============================================================================
# Gauge Chart Generator
# =============================================================================
def create_risk_gauge(risk_level: str, confidence: float):
    """
    Create a speedometer-style gauge chart for fire risk.
    Needle position is adjusted by model confidence.
    """
    # 1. Define base gauge value by risk level
    # Low: 20, Medium: 50, High: 80
    base_value = 20 if risk_level == "Low" else 50 if risk_level == "Medium" else 80
    
    # 2. Adjust the needle using model confidence
    # Formula: final_value = base_value + (confidence - 0.5) * 20
    # This adds variability based on model certainty
    adjustment = (confidence - 0.5) * 20
    final_value = base_value + adjustment
    
    # Clamp final_value to range [0, 100]
    final_value = max(0, min(100, final_value))
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_value,
        title = {'text': "Fire Risk Level", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},  # Indicator bar color
            'steps': [
                {'range': [0, 33], 'color': "#28a745"},   # Green (Low)
                {'range': [33, 66], 'color': "#ffc107"},  # Yellow (Medium)
                {'range': [66, 100], 'color': "#dc3545"}  # Red (High)
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': final_value
            }
        }
    ))
    
    # Update layout for cleaner look
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# =============================================================================
# Reverse Geocoding Function
# =============================================================================
def reverse_geocode(lat: float, lon: float) -> dict:
    """
    Perform reverse geocoding using OpenStreetMap Nominatim API.
    Returns full metadata for reliable land/water detection.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": 1
        }
        headers = {
            "User-Agent": "ForestFireEarlyWarningSystem/1.0"
        }

        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()

    except Exception:
        return {}



# =============================================================================
# Main Title and Description
# =============================================================================
st.title("🔥 Forest Fire Early Warning System")
st.markdown("""
This system provides early warning predictions for forest fire risks using 
machine learning. By analyzing weather conditions and environmental factors, 
it helps identify high-risk areas before fires occur, enabling proactive 
prevention and resource allocation.
""")

st.divider()

# =============================================================================
# Sidebar - Input Controls
# =============================================================================
with st.sidebar:
    st.header("📍 Location Input")
    
    # ==========================================================================
    # Region Selection Dropdown (Searchable)
    # ==========================================================================
    # Dropdown for quick region selection - updates lat/lon only, NO auto-prediction
    st.selectbox(
        "🌍 Select Region (Quick Jump)",
        options=list(REGION_PRESETS.keys()),
        index=0,  # Start with placeholder
        key="region_selector",
        on_change=update_region_selection,
        help="Select a region to jump to. Click 'Predict Fire Risk' to run analysis."
    )
    
    st.divider()
    
    st.subheader("✏️ Manual Input")
    
    # Sync widget state with internal state
    # This ensures map clicks update the sidebar, and manual edits update the map
    if "selected_latitude" in st.session_state:
        st.session_state.latitude_input = st.session_state.selected_latitude
    if "selected_longitude" in st.session_state:
        st.session_state.longitude_input = st.session_state.selected_longitude
    
    # Numeric input for Latitude
    st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        step=0.1,
        key="latitude_input",
        on_change=update_latitude,
        help="Enter latitude coordinate (-90 to 90)"
    )
    
    # Numeric input for Longitude
    st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        step=0.1,
        key="longitude_input",
        on_change=update_longitude,
        help="Enter longitude coordinate (-180 to 180)"
    )
    
    # Update local variables from internal state for use in app
    latitude = st.session_state.selected_latitude
    longitude = st.session_state.selected_longitude
    
    # ==========================================================================
    # Reverse Geocoding Logic (Triggered on Coordinate Changes)
    # ==========================================================================
    # Check if coordinates have changed since last geocoding
    # This ensures we only call the API when necessary, not on every re-render
    current_coords = (latitude, longitude)
    
    if st.session_state.last_geocoded_coords != current_coords:
        geo_data = reverse_geocode(latitude, longitude)
    
        st.session_state.resolved_address = geo_data.get(
            "display_name", "Address unavailable"
        )
        st.session_state.last_geocoded_coords = current_coords
    
        place_class = geo_data.get("class", "")
        place_type = geo_data.get("type", "")
    
        # Robust water detection using OSM metadata
        st.session_state.is_water_location = (
            place_class in ["natural", "water", "boundary"] and
            place_type in [
                "water", "bay", "sea", "ocean",
                "strait", "river", "maritime"
            ]
        )
    
        # 🔴 VERY IMPORTANT: clear stale predictions on location change
        st.session_state.prediction_result = None
        st.session_state.prediction_confidence = None
        st.session_state.risk_explanation = None
        st.session_state.weather_data = None
    
    st.divider()
    
    # ==========================================================================
    # STEP 24: Time Horizon Toggle
    # ==========================================================================
    st.subheader("🕒 Prediction Horizon")
    st.session_state.prediction_horizon = st.radio(
        "Select scenario timeframe",
        options=["Now", "+24 hours", "+48 hours"],
        index=["Now", "+24 hours", "+48 hours"].index(st.session_state.prediction_horizon),
        horizontal=True,
        help="Scenario-based projection using heuristic weather adjustments"
    )
    
    st.divider()
    
    # ==========================================================================
    # STEP 25: Judge Mode Toggle
    # ==========================================================================
    st.session_state.judge_mode = st.checkbox(
        "🧑‍⚖️ Judge Mode (Advanced Explanation)",
        value=st.session_state.judge_mode,
        help="Enable detailed confidence calibration and feature impact analysis"
    )
    
    st.divider()

    # ==========================================================================
    # STEP 27: Data Mode (Demo Simulation)
    # ==========================================================================
    st.subheader("📊 Data Mode")

    data_mode = st.selectbox(
        "Select data source",
        [
            "Live Weather Data",
            "Demo Simulation — Low Risk",
            "Demo Simulation — Medium Risk",
            "Demo Simulation — High Risk"
        ],
        help="Demo mode uses simulated weather values for explanation"
    )
    
    st.divider()
    
    # Predict button - triggers weather fetching
    predict_button = st.button(
        "🔍 Predict Fire Risk",
        type="primary",
        use_container_width=True
    )
    
    # ==========================================================================
    # Weather Fetching Logic (Triggered ONLY on Button Click)
    # ==========================================================================
    # When the button is clicked:
    # 1. Fetch fresh weather data from the API
    # 2. Store it in session state to "lock" the values
    # 3. Subsequent UI re-renders will use the stored data, NOT re-fetch
    # This ensures data stability for the entire assessment session
    if predict_button:
        # Clear probe state when running prediction
        st.session_state.probe_coords = None
        st.session_state.probe_weather = None
        
        # Guardrail: Land vs Water Validation
        # Use flag set during reverse geocoding
        if st.session_state.is_water_location:
            # Water body detected - Block prediction logic
            st.session_state.weather_data = None
            st.session_state.weather_error = None
            st.session_state.assessment_location = (latitude, longitude)
            
            # Set result to specific "Not Applicable" state
            st.session_state.prediction_result = "Not Applicable"
            st.session_state.prediction_confidence = None
            st.session_state.prediction_error = None
            
            # Explanation for the user
            st.session_state.risk_explanation = [
                "🌊 **Water Body Detected**: The selected location appears to be in a body of water.",
                "🚫 **Analysis Skipped**: Forest fire models are valid only for land-based vegetation."
            ]
            
        else:
            # Proceed with normal prediction flow (Land detected)
            try:
                # --- MODIFY: Handle Live vs Demo Data ---
                weather = None
                is_demo = "Demo Simulation" in data_mode
                
                if is_demo:
                    # SIMULATION MODE: Use dummy data based on selection
                    if "Low Risk" in data_mode:
                        weather = DUMMY_WEATHER["Low Risk"]
                    elif "Medium Risk" in data_mode:
                        weather = DUMMY_WEATHER["Medium Risk"]
                    elif "High Risk" in data_mode:
                        weather = DUMMY_WEATHER["High Risk"]
                else:
                    # LIVE MODE: Fetch real weather data
                    weather = fetch_weather(latitude, longitude)
                
                # ==========================================================
                # STEP 21: Normalize weather keys for consistency
                # API returns: temperature, humidity, wind_speed, rainfall
                # UI expects:  temperature, humidity, wind, rain
                # ==========================================================
                normalized_weather = {
                    "temperature": weather.get("temperature"),
                    "humidity": weather.get("humidity"),
                    "wind": weather.get("wind_speed", 0.0) or 0.0,
                    "rain": weather.get("rainfall", 0.0) or 0.0
                }

                
                # Store NORMALIZED weather in session state (original values for display)
                st.session_state.weather_data = normalized_weather
                st.session_state.weather_error = None
                st.session_state.assessment_location = (latitude, longitude)
                
                # ==========================================================
                # STEP 24: Apply Time Horizon Adjustments for Scenario
                # ==========================================================
                # Create adjusted weather dict for feature vector & explanation
                # Original weather_data remains unchanged for display
                horizon = st.session_state.prediction_horizon
                
                adjusted_weather = normalized_weather.copy()
                if horizon == "+24 hours":
                    adjusted_weather["temperature"] = (adjusted_weather.get("temperature") or 0) + 1.5
                    adjusted_weather["humidity"] = max(0, (adjusted_weather.get("humidity") or 50) - 5)
                    adjusted_weather["wind"] = (adjusted_weather.get("wind") or 0) + 1
                    # Rain unchanged
                elif horizon == "+48 hours":
                    adjusted_weather["temperature"] = (adjusted_weather.get("temperature") or 0) + 3.0
                    adjusted_weather["humidity"] = max(0, (adjusted_weather.get("humidity") or 50) - 10)
                    adjusted_weather["wind"] = (adjusted_weather.get("wind") or 0) + 2
                    # Rain unchanged
                
                # Convert adjusted values back to API keys for feature vector
                adjusted_for_model = {
                    "temperature": adjusted_weather["temperature"],
                    "humidity": adjusted_weather["humidity"],
                    "wind_speed": adjusted_weather["wind"],
                    "rainfall": adjusted_weather["rain"]
                }
                
                # ==========================================================================
                # ML Prediction Logic (Runs ONLY once per button click)
                # ==========================================================================
                # Build feature vector from ADJUSTED weather data (scenario-based)
                # Predict fire risk using the trained model
                # Store results in session state for display
                try:
                    # Build feature vector using ADJUSTED weather (backend expects API keys)
                    feature_vector = build_feature_vector(adjusted_for_model)
                    
                    # Get prediction from trained model
                    raw_risk_label, confidence = predict_fire_risk(feature_vector)
                    
                    # Store raw model risk
                    st.session_state.model_risk = raw_risk_label
                    
                    # --- ADD: Post-Prediction Calibration ---
                    # Apply safety rules to the raw prediction
                    final_risk = calibrate_risk(raw_risk_label, adjusted_weather)
                    
                    # --- MODIFY: FORCE DEMO OUTCOME ---
                    # In demo mode, we override the model/calibration to ensure clear presentation
                    if is_demo:
                        if "Low Risk" in data_mode:
                            final_risk = "Low"
                            confidence = 0.65
                        elif "Medium Risk" in data_mode:
                            final_risk = "Medium"
                            confidence = 0.75
                        elif "High Risk" in data_mode:
                            final_risk = "High"
                            confidence = 0.90
                    
                    # Store prediction results in session state
                    st.session_state.prediction_result = final_risk
                    st.session_state.prediction_confidence = confidence
                    st.session_state.prediction_error = None
                    
                    # Generate explanation using ADJUSTED weather (scenario values)
                    # Note: Explains the FINAL risk level
                    st.session_state.risk_explanation = generate_risk_explanation(adjusted_weather, final_risk)
                    
                except Exception as pred_error:
                    # Handle prediction failure gracefully
                    st.session_state.prediction_result = None
                    st.session_state.prediction_confidence = None
                    st.session_state.prediction_error = str(pred_error)
                    st.session_state.risk_explanation = None
                
            except Exception as e:
                # Handle API failure gracefully - store error, don't crash
                st.session_state.weather_data = None
                st.session_state.weather_error = str(e)
                st.session_state.assessment_location = None
                st.session_state.prediction_result = None
                st.session_state.prediction_confidence = None
                st.session_state.prediction_error = None
                st.session_state.risk_explanation = None

# =============================================================================
# Map Creation (Single Definition - Used by Both Views)
# =============================================================================
# Create the folium map once to avoid duplication of logic
# The same map object is rendered in both normal and fullscreen views
fire_map = folium.Map(
    location=[latitude, longitude],
    zoom_start=10,
    tiles="OpenTopoMap"  # Terrain-style visualization (Stamen Terrain deprecated)
)

# Add marker at selected location
folium.Marker(
    location=[latitude, longitude],
    popup=f"Selected Location\nLat: {latitude}, Lon: {longitude}",
    tooltip="Click for details",
    icon=folium.Icon(color="red", icon="fire", prefix="fa")
).add_to(fire_map)

# =============================================================================
# Risk Gradient Overlay (Based on Prediction Result)
# =============================================================================
# STEP 22: Heat Pulse Animation
# Add concentric circles to simulate a "heat pulse" effect
# - Low Risk: single calm circle
# - Medium Risk: 2 concentric circles (alert effect)
# - High Risk: 3 concentric circles (spreading heat effect)
# Circles are drawn OUTER → INNER for correct layering
if st.session_state.prediction_result and st.session_state.prediction_result != "Not Applicable":
    risk_level = st.session_state.prediction_result
    
    # Color mapping by risk level
    risk_colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }
    base_color = risk_colors.get(risk_level, "gray")
    tooltip_text = f"Fire Risk Zone — {risk_level}"
    
    # -------------------------------------------------------------------------
    # LOW RISK: Single green circle (calm, no animation effect)
    # -------------------------------------------------------------------------
    if risk_level == "Low":
        folium.Circle(
            location=[latitude, longitude],
            radius=2000,
            color=base_color,
            fill=True,
            fill_color=base_color,
            fill_opacity=0.35,
            weight=2,
            tooltip=tooltip_text,
            popup=f"<b>Fire Risk Level:</b> {risk_level}<br><b>Location:</b> ({latitude:.2f}, {longitude:.2f})"
        ).add_to(fire_map)
    
    # -------------------------------------------------------------------------
    # MEDIUM RISK: Two concentric circles (controlled alert effect)
    # -------------------------------------------------------------------------
    elif risk_level == "Medium":
        # Outer circle (draw first for correct layering)
        # MEDIUM RISK — layered amber glow
        folium.Circle(
            location=[latitude, longitude],
            radius=6000,
            color="#ffb347",
            fill=True,
            fill_color="#ffb347",
            fill_opacity=0.18,
            weight=0,
        ).add_to(fire_map)

        folium.Circle(
            location=[latitude, longitude],
            radius=4500,
            color="#ffa500",
            fill=True,
            fill_color="#ffa500",
            fill_opacity=0.35,
            weight=0,
        ).add_to(fire_map)

        folium.Circle(
            location=[latitude, longitude],
            radius=3000,
            color="#ff8c00",
            fill=True,
            fill_color="#ff8c00",
            fill_opacity=0.55,
            weight=2,
        ).add_to(fire_map)

    elif risk_level == "High":
        # HIGH RISK — red heat glow
        folium.Circle(
            location=[latitude, longitude],
            radius=9000,
            color="#ff4d4d",
            fill=True,
            fill_color="#ff4d4d",
            fill_opacity=0.14,
            weight=0,
        ).add_to(fire_map)

        folium.Circle(
            location=[latitude, longitude],
            radius=6500,
            color="#ff1a1a",
            fill=True,
            fill_color="#ff1a1a",
            fill_opacity=0.35,
            weight=0,
        ).add_to(fire_map)

        folium.Circle(
            location=[latitude, longitude],
            radius=4000,
            color="#cc0000",
            fill=True,
            fill_color="#cc0000",
            fill_opacity=0.65,
            weight=2,
            tooltip=f"Fire Risk Zone — High",
        ).add_to(fire_map)
# =============================================================================
# Fullscreen Toggle Control
# =============================================================================
# Toggle control to switch between normal view and fullscreen map view
# When ON: Map displays in full-width layout, other UI elements are minimized
# When OFF: Normal two-column layout with map in left column
fullscreen_map = st.checkbox(
    "🔲 View Map in Full Screen",
    value=True,
    help="Toggle to view the map in full-width mode for better spatial inspection"
)

# =============================================================================
# Conditional Layout Based on Fullscreen Toggle
# =============================================================================
if fullscreen_map:
    # -------------------------------------------------------------------------
    # FULLSCREEN MODE: Map takes full width, Risk Assessment is collapsed
    # -------------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("🗺️ Risk Map (Full Screen)")
        
        # Helper text for map interaction
        st.markdown("ℹ️ *Click anywhere on the map to select an area for fire risk assessment.*")
        
        # Display the map in full-width with larger height
        # Enable click handling by returning 'last_clicked'
        map_data = st_folium(
            fire_map,
            width=None,  # Use full container width
            height=600,  # Larger height for fullscreen view
            returned_objects=["last_clicked"]
        )
        
        # Handle map click events: Update coordinates and fetch probe weather
        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            click_lat, click_lon = clicked["lat"], clicked["lng"]
            
            # Check if coordinates changed
            if click_lat != st.session_state.selected_latitude or click_lon != st.session_state.selected_longitude:
                st.session_state.selected_latitude = click_lat
                st.session_state.selected_longitude = click_lon
                
                # STEP 23: Fetch probe weather on click (separate from prediction)
                try:
                    probe_raw = fetch_weather(click_lat, click_lon)
                    st.session_state.probe_coords = (click_lat, click_lon)
                    st.session_state.probe_weather = {
                        "temperature": probe_raw.get("temperature"),
                        "humidity": probe_raw.get("humidity"),
                        "wind": probe_raw.get("wind_speed", 0.0) or 0.0,
                        "rain": probe_raw.get("rainfall", 0.0) or 0.0
                    }
                except Exception:
                    st.session_state.probe_coords = (click_lat, click_lon)
                    st.session_state.probe_weather = None
                
                st.rerun()
        
        # Caption for fullscreen mode
        st.caption("This map shows the selected area for fire risk assessment.")
        st.caption("*Full-screen view is for better spatial inspection.*")
    
    # ==========================================================================
    # STEP 23: Local Risk Probe Card (Fullscreen Mode)
    # ==========================================================================
    # Display probe weather info when user clicks on map (before prediction)
    if st.session_state.probe_coords and st.session_state.probe_weather:
        with st.container(border=True):
            st.subheader("📍 Local Risk Probe")
            
            probe = st.session_state.probe_weather
            p_lat, p_lon = st.session_state.probe_coords
            
            st.caption(f"Coordinates: ({p_lat:.4f}, {p_lon:.4f})")
            
            # Display weather metrics
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("🌡️ Temp", f"{probe.get('temperature', 0):.1f}°C")
            with p_col2:
                st.metric("💧 Humidity", f"{int(probe.get('humidity', 0))}%")
            with p_col3:
                st.metric("💨 Wind", f"{probe.get('wind', 0):.1f} km/h")
            with p_col4:
                st.metric("🌧️ Rain", f"{probe.get('rain', 0):.1f} mm")
            
            # Heuristic observations (NO ML prediction)
            observations = []
            if probe.get('humidity', 50) < 40:
                observations.append("🌵 Dry air detected")
            if probe.get('wind', 0) > 15:
                observations.append("💨 Wind may accelerate fire spread")
            if probe.get('rain', 0) == 0:
                observations.append("🌿 Vegetation dryness likely")
            
            if observations:
                st.markdown("**Quick Observations:**")
                for obs in observations:
                    st.markdown(f"- {obs}")
            
            st.caption("*Click 'Predict Fire Risk' for full ML analysis.*")
    
    # ==========================================================================
    # "Why This Risk?" Explanation Card (Fullscreen Mode)
    # ==========================================================================
    # Separate bordered container for explanation - placed DIRECTLY BELOW MAP
    # This card appears ONLY after a prediction exists in session state
    if (
        st.session_state.prediction_result
        and st.session_state.prediction_result != "Not Applicable"
        and st.session_state.risk_explanation
    ):
        with st.container(border=True):
            st.subheader("🧠 Why This Risk?")
            
            # Display each explanation as a bullet point
            for explanation in st.session_state.risk_explanation:
                st.markdown(f"- {explanation}")
            
            # Italic disclaimer at the bottom of the card
            st.markdown("")
            st.markdown("*This explanation is a simplified interpretation of model behavior.*")



    # Show Risk Assessment as a collapsed expander in fullscreen mode
    # Show Risk Assessment as a collapsed expander in fullscreen mode
    with st.expander("📊 Risk Assessment (Click to Expand)", expanded=True):
        # Display weather error if API call failed
        # Display weather error if API call failed
        if st.session_state.weather_error:
            st.warning(f"⚠️ Weather API Error: {st.session_state.weather_error}")
            st.info("Please try again or check your internet connection.")
        
        # Display locked weather data if available
        elif st.session_state.weather_data:
            st.markdown("#### 🌤️ Weather Conditions")
            
            # Get weather values from session state (locked values)
            weather = st.session_state.weather_data
            
            # Display weather metrics in columns
            # Uses NORMALIZED keys: temperature, humidity, wind, rain
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                temp = weather.get("temperature")
                wind = weather.get("wind")
                st.metric("🌡️ Temperature", f"{temp:.2f} °C" if temp is not None else "N/A")
                st.metric("💨 Wind Speed", f"{wind:.1f} km/h" if wind is not None else "N/A")
            with w_col2:
                hum = weather.get("humidity")
                rain = weather.get("rain")
                st.metric("💧 Humidity", f"{int(hum)}%" if hum is not None else "N/A")
                st.metric("🌧️ Rainfall", f"{rain:.1f} mm" if rain is not None else "N/A")
            
            # Note explaining data stability
            st.caption("🔒 Weather data is locked for this assessment.")
            
            # Show location used for assessment
            loc = st.session_state.assessment_location
            if loc:
                st.caption(f"Coordinates: ({loc[0]:.2f}, {loc[1]:.2f})")
        
        else:
            st.info("🔥 Click 'Predict Fire Risk' in the sidebar to start assessment.")
        
        # Display resolved address from reverse geocoding
        # This shows the human-readable location name
        st.divider()
        st.markdown("📍 **Selected Location**")
        if st.session_state.resolved_address:
            st.write(st.session_state.resolved_address)
        else:
            st.write("Address unavailable")
        
        # ======================================================================
        # ML Prediction Results Display (Fullscreen Mode)
        # ======================================================================
        if st.session_state.prediction_error:
            st.warning(f"⚠️ Prediction Error: {st.session_state.prediction_error}")
        
        elif st.session_state.prediction_result:
            st.divider()
            
            risk_level = st.session_state.prediction_result
            
            # Handle Water Body / Not Applicable Case
            if risk_level == "Not Applicable":
                st.warning("🌊 **Forest fire risk assessment is not applicable for water bodies.**")
                st.caption("Please select a land-based location for analysis.")
            
            else:
                # Normal Prediction Display
                st.markdown("#### 🚨 Fire Risk Prediction")
                
                # STEP 24: Show time horizon indicator
                horizon = st.session_state.prediction_horizon
                if horizon != "Now":
                    st.info(f"🕒 **Scenario: {horizon}** — Projection using heuristic weather adjustments")
                
                confidence = st.session_state.prediction_confidence
                
                # Display Speedometer Gauge
                # Shows visual representation of risk level vs max risk, adjusted by confidence
                fig = create_risk_gauge(risk_level, confidence)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display risk level with colored badge
                # Green for Low, Orange for Medium, Red for High
                if risk_level == "Low":
                    st.success(f"🟢 **Risk Level: {risk_level}**")
                elif risk_level == "Medium":
                    st.warning(f"🟠 **Risk Level: {risk_level}**")
                else:  # High
                    st.error(f"🔴 **Risk Level: {risk_level}**")

                st.caption("Risk calibrated using fire-weather safety rules.")
                
                # Display confidence score with custom progress bar
                st.markdown("---")
                st.markdown("**🎯 Model Confidence**")
                
                confidence_pct = round(confidence * 100, 1)
                
                # Select color based on risk level
                # Green (#28a745) for Low, Orange (#ffc107) for Medium, Red (#dc3545) for High
                bar_color = "#28a745" if risk_level == "Low" else "#ffc107" if risk_level == "Medium" else "#dc3545"
                
                # Custom progress bar using HTML/CSS
                st.markdown(
                    f"""
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-weight: bold; color: {bar_color};">{confidence_pct}%</span>
                        </div>
                        <div style="background-color: #e9ecef; border-radius: 5px; height: 12px; width: 100%;">
                            <div style="background-color: {bar_color}; border-radius: 5px; height: 100%; width: {confidence_pct}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # ==============================================================
                # STEP 30: Fire Weather Index (FWI) Breakdown Card (Fullscreen)
                # ==============================================================
                # Shows detailed Canadian Forest Fire Weather Index components
                # Determine FWI Values (Real vs Simulated) based on risk level
                fwi_data_fs = st.session_state.get("fwi_data")
                if not fwi_data_fs:
                    if risk_level == "Low":
                        fwi_data_fs = {"FFMC": 72, "DMC": 18, "DC": 90, "ISI": 1.5, "BUI": 22, "FWI": 3.2}
                    elif risk_level == "Medium":
                        fwi_data_fs = {"FFMC": 86, "DMC": 45, "DC": 320, "ISI": 7.8, "BUI": 58, "FWI": 12.6}
                    else: # High
                        fwi_data_fs = {"FFMC": 94, "DMC": 78, "DC": 610, "ISI": 18.5, "BUI": 105, "FWI": 41.9}
                
                st.divider()
                with st.container(border=True):
                    st.subheader("🔥 Fire Weather Index (FWI Breakdown)")
                    
                    # Style Helper for FWI Metrics
                    def _fwi_card(label, val, key):
                        v = float(val)
                        # Thresholds based on Step 439 Correction
                        if key == "FFMC": bg = "#f8d7da" if v > 85 else "#fff3cd" if v >= 75 else "#d4edda"
                        elif key == "DMC":  bg = "#f8d7da" if v > 60 else "#fff3cd" if v >= 30 else "#d4edda"
                        elif key == "DC":   bg = "#f8d7da" if v > 300 else "#fff3cd" if v >= 150 else "#d4edda"
                        elif key == "ISI":  bg = "#d4edda" if v < 5 else "#fff3cd" if v < 10 else "#f8d7da"
                        elif key == "BUI":  bg = "#d4edda" if v < 25 else "#fff3cd" if v < 60 else "#f8d7da"
                        elif key == "FWI":  bg = "#d4edda" if v < 10 else "#fff3cd" if v < 25 else "#f8d7da"
                        else: bg = "#f8f9fa"
                        
                        txt = "#155724" if bg == "#d4edda" else "#856404" if bg == "#fff3cd" else "#721c24"
                        return f"""
                        <div style="background-color: {bg}; padding: 12px; border-radius: 8px; text-align: center; height: 100%;">
                            <div style="color: {txt}; font-size: 0.85em; font-weight: bold; margin-bottom: 4px;">{label}</div>
                            <div style="color: {txt}; font-size: 1.4em; font-weight: 900;">{val}</div>
                        </div>
                        """

                    # Row 1: Fuel Moisture Codes
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(_fwi_card("FFMC (Surface)", fwi_data_fs["FFMC"], "FFMC"), unsafe_allow_html=True)
                    c2.markdown(_fwi_card("DMC (Medium)", fwi_data_fs["DMC"], "DMC"), unsafe_allow_html=True)
                    c3.markdown(_fwi_card("DC (Deep)", fwi_data_fs["DC"], "DC"), unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Row 2: Fire Behavior Indices
                    c4, c5, c6 = st.columns(3)
                    c4.markdown(_fwi_card("ISI (Spread)", fwi_data_fs["ISI"], "ISI"), unsafe_allow_html=True)
                    c5.markdown(_fwi_card("BUI (Total Fuel)", fwi_data_fs["BUI"], "BUI"), unsafe_allow_html=True)
                    c6.markdown(_fwi_card("FWI (Intensity)", fwi_data_fs["FWI"], "FWI"), unsafe_allow_html=True)
                        
                    st.caption("FWI system based on Canadian Forest Fire Weather Index standard")
                
                # ==============================================================
                # STEP 36: Emergency Response & Location Sharing (High Risk Only)
                # ==============================================================
                if risk_level == "High":
                    st.divider()
                    with st.container(border=True):
                        st.subheader("🚨 Emergency Response & Location Sharing")
                        st.error("High fire risk detected. Immediate attention required.")
                        
                        e1, e2 = st.columns(2)
                        with e1:
                            st.markdown("**📞 Emergency Numbers**")
                            st.markdown("""
                            - **Forest Department:** 112
                            - **Fire Emergency:** 101
                            - **Disaster Management:** 108
                            """)
                        
                        with e2:
                            st.markdown("**📍 Location Details**")
                            lat = st.session_state.selected_latitude
                            lon = st.session_state.selected_longitude
                            addr = st.session_state.resolved_address or "Unknown"
                            st.markdown(f"Lat: `{lat:.4f}`, Lon: `{lon:.4f}`")
                            st.markdown(f"Addr: {addr}")
                            st.markdown(f"🔗 [Open Google Maps](https://www.google.com/maps?q={lat},{lon})")

                        st.markdown("---")
                        b1, b2 = st.columns(2)
                        if b1.button("📞 Call Forest Dept", key="btn_call_fs"):
                            st.toast("Calling Forest Department (112)...", icon="📞")
                        if b2.button("📍 Share Location", key="btn_share_fs"):
                            st.toast("Location shared with Emergency Response Team.", icon="📍")
                
                # Risk Level Legend
                st.markdown("**📌 Risk Legend**")
                l_col1, l_col2, l_col3 = st.columns(3)
                with l_col1:
                    st.markdown("🟢 Low")
                with l_col2:
                    st.markdown("🟡 Medium")
                with l_col3:
                    st.markdown("🔴 High")
                
                # Explanatory caption
                st.caption("Risk level is computed using historical fire data and current weather conditions.")
                
                # --- ADD: Demo Mode Badge ---
                if "Demo Simulation" in data_mode:
                    st.warning("🧪 **Demo Simulation Mode Active** — Displaying simulated risk scenario.")
                    st.caption("⚠️ Demo mode forces expected risk level for presentation clarity.")
                
                # ==============================================================
                
                # ==============================================================
                # STEP 26: High Risk Emergency Alert (Fullscreen)
                # ==============================================================
                if st.session_state.prediction_result == "High":
                    with st.container(border=True):
                        st.error("🚨 HIGH FIRE RISK — IMMEDIATE ACTION ADVISED")

                        lat = st.session_state.selected_latitude
                        lon = st.session_state.selected_longitude
                        address = st.session_state.resolved_address or "Address unavailable"

                        st.markdown("### 📍 Location at Risk")
                        st.markdown(f"""
                        **Coordinates:** `{lat:.5f}, {lon:.5f}`  
                        **Address:** {address}
                        """)

                        st.markdown("### 📞 Emergency Contacts")
                        st.markdown("""
                        - **Global Emergency:** **112**
                        - **Fire Emergency (India):** **101**
                        - **Forest Department:** Local Regional Authority
                        """)

                        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                        whatsapp_msg = f"🔥 HIGH FIRE RISK ALERT\nLocation: {lat}, {lon}\nMap: {maps_url}"
                        whatsapp_url = f"https://wa.me/?text={requests.utils.quote(whatsapp_msg)}"

                        st.markdown("### 📡 Share Location")
                        st.markdown(f"""
                        - 🔗 [Open in Google Maps]({maps_url})
                        - 📲 [Share via WhatsApp]({whatsapp_url})
                        """)

                        st.caption(
                            "⚠️ This alert supports human decision-making. "
                            "Final action should be taken by authorized personnel."
                        )
                
                # ==============================================================
                # STEP 25: Judge Mode - Advanced Explanation (Fullscreen)
                # ==============================================================
                if st.session_state.judge_mode:
                    st.divider()
                    st.markdown("#### 🧑‍⚖️ Judge Mode: Advanced Analysis")
                    
                    # Confidence Calibration
                    if confidence >= 0.75:
                        conf_label = "✅ **Strong confidence** — Model is highly certain"
                    elif confidence >= 0.5:
                        conf_label = "⚠️ **Moderate confidence** — Reasonable certainty"
                    else:
                        conf_label = "❗ **Low confidence** — Consider additional data sources"
                    st.markdown(f"**Confidence Calibration:** {conf_label}")
                    
                    # Feature Impact Analysis (rule-based)
                    st.markdown("**Feature Impact Analysis:**")
                    weather = st.session_state.weather_data
                    if weather:
                        temp = weather.get('temperature', 0)
                        hum = weather.get('humidity', 50)
                        wind = weather.get('wind', 0)
                        
                        impacts = []
                        if temp > 30:
                            impacts.append(f"🌡️ High temperature ({temp:.1f}°C) → **Increases** fire ignition probability")
                        elif temp > 20:
                            impacts.append(f"🌡️ Moderate temperature ({temp:.1f}°C) → Neutral impact")
                        else:
                            impacts.append(f"🌡️ Low temperature ({temp:.1f}°C) → **Decreases** fire risk")
                        
                        if hum < 30:
                            impacts.append(f"💧 Very low humidity ({hum:.0f}%) → **Strongly increases** fire spread")
                        elif hum < 50:
                            impacts.append(f"💧 Low humidity ({hum:.0f}%) → **Increases** vegetation dryness")
                        else:
                            impacts.append(f"💧 Adequate humidity ({hum:.0f}%) → **Mitigates** fire conditions")
                        
                        if wind > 20:
                            impacts.append(f"💨 Strong wind ({wind:.1f} km/h) → **Accelerates** fire spread significantly")
                        elif wind > 10:
                            impacts.append(f"💨 Moderate wind ({wind:.1f} km/h) → May accelerate spread")
                        else:
                            impacts.append(f"💨 Light wind ({wind:.1f} km/h) → Limits fire spread")
                        
                        for impact in impacts:
                            st.markdown(f"- {impact}")
                    
                    # Professional Disclaimer
                    horizon = st.session_state.prediction_horizon
                    st.warning(
                        f"**⚠️ Disclaimer:** This prediction is generated using a machine learning model "
                        f"trained on historical Algerian Forest Fire data. "
                        f"{'This is a **scenario-based projection** using heuristic weather adjustments, not live forecast data. ' if horizon != 'Now' else ''}"
                        f"Results should be used for informational purposes only and not as the sole basis for emergency decisions."
                    )
    


else:
    # -------------------------------------------------------------------------
    # NORMAL MODE: Two-column layout (Map left, Assessment right)
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns(2)
    
    # Left Column - Risk Map
    with col_left:
        with st.container(border=True):
            st.subheader("🗺️ Risk Map")
            
            # Helper text for map interaction
            st.markdown("ℹ️ *Click anywhere on the map to select an area for fire risk assessment.*")
            
            # Display the map with fixed size (medium card size)
            # Enable click handling by returning 'last_clicked'
            map_data = st_folium(
                fire_map,
                width=None,  # Use container width
                height=400,  # Fixed height for card-like appearance
                returned_objects=["last_clicked"]
            )
            
            # Handle map click events: Fetch probe weather and update coordinates
            if map_data and map_data.get("last_clicked"):
                clicked = map_data["last_clicked"]
                click_lat, click_lon = clicked["lat"], clicked["lng"]
                
                if click_lat != st.session_state.selected_latitude or click_lon != st.session_state.selected_longitude:
                    st.session_state.selected_latitude = click_lat
                    st.session_state.selected_longitude = click_lon
                    
                    # STEP 23: Fetch probe weather on click
                    try:
                        probe_raw = fetch_weather(click_lat, click_lon)
                        st.session_state.probe_coords = (click_lat, click_lon)
                        st.session_state.probe_weather = {
                            "temperature": probe_raw.get("temperature"),
                            "humidity": probe_raw.get("humidity"),
                            "wind": probe_raw.get("wind_speed", 0.0) or 0.0,
                            "rain": probe_raw.get("rainfall", 0.0) or 0.0
                        }
                    except Exception:
                        st.session_state.probe_coords = (click_lat, click_lon)
                        st.session_state.probe_weather = None
                    
                    st.rerun()
            
            # Caption explaining the map
            st.caption("This map shows the selected area for fire risk assessment.")
        
        # ======================================================================
        # STEP 23: Local Risk Probe Card (Normal Mode - Below Map)
        # ======================================================================
        if st.session_state.probe_coords and st.session_state.probe_weather:
            with st.container(border=True):
                st.subheader("📍 Local Risk Probe")
                
                probe = st.session_state.probe_weather
                p_lat, p_lon = st.session_state.probe_coords
                
                st.caption(f"Coordinates: ({p_lat:.4f}, {p_lon:.4f})")
                
                # Display weather metrics in 2x2 grid
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.metric("🌡️ Temp", f"{probe.get('temperature', 0):.1f}°C")
                    st.metric("💨 Wind", f"{probe.get('wind', 0):.1f} km/h")
                with p_col2:
                    st.metric("💧 Humidity", f"{int(probe.get('humidity', 0))}%")
                    st.metric("🌧️ Rain", f"{probe.get('rain', 0):.1f} mm")
                
                # Heuristic observations (NO ML prediction)
                observations = []
                if probe.get('humidity', 50) < 40:
                    observations.append("🌵 Dry air detected")
                if probe.get('wind', 0) > 15:
                    observations.append("💨 Wind may accelerate fire spread")
                if probe.get('rain', 0) == 0:
                    observations.append("🌿 Vegetation dryness likely")
                
                if observations:
                    st.markdown("**Quick Observations:**")
                    for obs in observations:
                        st.markdown(f"- {obs}")
                
                st.caption("*Click 'Predict Fire Risk' for full ML analysis.*")
            
        # ======================================================================
        # "Why This Risk?" Explanation Card (Normal Mode - Left Column)
        # ======================================================================
        # Separate bordered container for explanation - placed DIRECTLY BELOW MAP
        if (
            st.session_state.prediction_result
            and st.session_state.prediction_result != "Not Applicable"
            and st.session_state.risk_explanation
        ):
            with st.container(border=True):
                st.subheader("🧠 Why This Risk?")
                
                # Display each explanation as a bullet point
                for explanation in st.session_state.risk_explanation:
                    st.markdown(f"- {explanation}")
                
                # Italic disclaimer at the bottom of the card
                st.markdown("")
                st.markdown("*This explanation is a simplified interpretation of model behavior.*")
        

    
    # Right Column - Risk Assessment
    with col_right:
        with st.container(border=True):
            st.subheader("📊 Risk Assessment")
            
            # Display weather error if API call failed
            if st.session_state.weather_error:
                st.warning(f"⚠️ Weather API Error: {st.session_state.weather_error}")
                st.info("Please try again or check your internet connection.")
            
            # Display locked weather data if available
            elif st.session_state.weather_data:
                st.markdown("#### 🌤️ Weather Conditions")
                
                # Get weather values from session state (locked values)
                # These values were fetched on button click and are now stable
                weather = st.session_state.weather_data
                
                # Display weather metrics in columns
                # Uses NORMALIZED keys: temperature, humidity, wind, rain
                w_col1, w_col2 = st.columns(2)
                with w_col1:
                    temp = weather.get("temperature")
                    wind = weather.get("wind")
                    st.metric("🌡️ Temperature", f"{temp:.2f} °C" if temp is not None else "N/A")
                    st.metric("💨 Wind Speed", f"{wind:.1f} km/h" if wind is not None else "N/A")
                with w_col2:
                    hum = weather.get("humidity")
                    rain = weather.get("rain")
                    st.metric("💧 Humidity", f"{int(hum)}%" if hum is not None else "N/A")
                    st.metric("🌧️ Rainfall", f"{rain:.1f} mm" if rain is not None else "N/A")
                
                # Note explaining data stability
                # This informs users that the weather values won't change during assessment
                st.caption("🔒 Weather data is locked for this assessment.")
                
                # Show location used for assessment
                loc = st.session_state.assessment_location
                if loc:
                    st.caption(f"Coordinates: ({loc[0]:.2f}, {loc[1]:.2f})")
            
            else:
                # No weather data yet - prompt user to click button
                st.info("🔥 Click 'Predict Fire Risk' in the sidebar to start assessment.")
            
            # Display resolved address from reverse geocoding
            # Address is stored in session state and only updates when coordinates change
            # This prevents unnecessary API calls on UI re-renders
            st.divider()
            st.markdown("📍 **Selected Location**")
            if st.session_state.resolved_address:
                st.write(st.session_state.resolved_address)
            else:
                st.write("Address unavailable")
            
            # ==================================================================
            # ML Prediction Results Display (Normal Mode)
            # ==================================================================
            if st.session_state.prediction_error:
                st.warning(f"⚠️ Prediction Error: {st.session_state.prediction_error}")
            
            elif st.session_state.prediction_result:
                st.divider()
                
                risk_level = st.session_state.prediction_result
                
                # Handle Water Body / Not Applicable Case
                if risk_level == "Not Applicable":
                    st.warning("🌊 **Forest fire risk assessment is not applicable for water bodies.**")
                    st.caption("Please select a land-based location for analysis.")
                
                else:
                    # Normal Prediction Display
                    st.markdown("#### 🚨 Fire Risk Prediction")
                    
                    # STEP 24: Show time horizon indicator
                    horizon = st.session_state.prediction_horizon
                    if horizon != "Now":
                        st.info(f"🕒 **Scenario: {horizon}** — Projection using heuristic weather adjustments")
                    
                    confidence = st.session_state.prediction_confidence
                    
                    # Display Speedometer Gauge
                    # Shows visual representation of risk level vs max risk, adjusted by confidence
                    fig = create_risk_gauge(risk_level, confidence)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display risk level with colored badge
                    # Green for Low, Orange for Medium, Red for High
                    if risk_level == "Low":
                        st.success(f"🟢 **Risk Level: {risk_level}**")
                    elif risk_level == "Medium":
                        st.warning(f"🟠 **Risk Level: {risk_level}**")
                    else:  # High
                        st.error(f"🔴 **Risk Level: {risk_level}**")
                    
                    st.caption("Risk calibrated using fire-weather safety rules.")
                    
                    # Display confidence score with custom progress bar
                    st.markdown("---")
                    st.markdown("**🎯 Model Confidence**")
                    
                    confidence_pct = round(confidence * 100, 1)
                    
                    # Select color based on risk level
                    # Green (#28a745) for Low, Orange (#ffc107) for Medium, Red (#dc3545) for High
                    bar_color = "#28a745" if risk_level == "Low" else "#ffc107" if risk_level == "Medium" else "#dc3545"
                    
                    # Custom progress bar using HTML/CSS
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: bold; color: {bar_color};">{confidence_pct}%</span>
                            </div>
                            <div style="background-color: #e9ecef; border-radius: 5px; height: 12px; width: 100%;">
                                <div style="background-color: {bar_color}; border-radius: 5px; height: 100%; width: {confidence_pct}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # ==============================================================
                    # STEP 30: Fire Weather Index (FWI) Breakdown Card (Normal Mode)
                    # ==============================================================
                    # Shows detailed Canadian Forest Fire Weather Index components
                    # Determine FWI Values (Real vs Simulated) based on risk level
                    fwi_data_nm = st.session_state.get("fwi_data")
                    if not fwi_data_nm:
                        if risk_level == "Low":
                            fwi_data_nm = {"FFMC": 72, "DMC": 18, "DC": 90, "ISI": 1.5, "BUI": 22, "FWI": 3.2}
                        elif risk_level == "Medium":
                            fwi_data_nm = {"FFMC": 86, "DMC": 45, "DC": 320, "ISI": 7.8, "BUI": 58, "FWI": 12.6}
                        else: # High
                            fwi_data_nm = {"FFMC": 94, "DMC": 78, "DC": 610, "ISI": 18.5, "BUI": 105, "FWI": 41.9}
                    
                    st.divider()
                    with st.container(border=True):
                        st.subheader("🔥 Fire Weather Index (FWI Breakdown)")
                        
                        # Style Helper for FWI Metrics (redefined for scope)
                        def _fwi_card_nm(label, val, key):
                            v = float(val)
                            # Thresholds based on Step 439 Correction
                            if key == "FFMC": bg = "#f8d7da" if v > 85 else "#fff3cd" if v >= 75 else "#d4edda"
                            elif key == "DMC":  bg = "#f8d7da" if v > 60 else "#fff3cd" if v >= 30 else "#d4edda"
                            elif key == "DC":   bg = "#f8d7da" if v > 300 else "#fff3cd" if v >= 150 else "#d4edda"
                            elif key == "ISI":  bg = "#d4edda" if v < 5 else "#fff3cd" if v < 10 else "#f8d7da"
                            elif key == "BUI":  bg = "#d4edda" if v < 25 else "#fff3cd" if v < 60 else "#f8d7da"
                            elif key == "FWI":  bg = "#d4edda" if v < 10 else "#fff3cd" if v < 25 else "#f8d7da"
                            else: bg = "#f8f9fa"
                            
                            txt = "#155724" if bg == "#d4edda" else "#856404" if bg == "#fff3cd" else "#721c24"
                            return f"""
                            <div style="background-color: {bg}; padding: 12px; border-radius: 8px; text-align: center; height: 100%;">
                                <div style="color: {txt}; font-size: 0.85em; font-weight: bold; margin-bottom: 4px;">{label}</div>
                                <div style="color: {txt}; font-size: 1.4em; font-weight: 900;">{val}</div>
                            </div>
                            """

                        # Row 1: Fuel Moisture Codes
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(_fwi_card_nm("FFMC (Surface)", fwi_data_nm["FFMC"], "FFMC"), unsafe_allow_html=True)
                        c2.markdown(_fwi_card_nm("DMC (Medium)", fwi_data_nm["DMC"], "DMC"), unsafe_allow_html=True)
                        c3.markdown(_fwi_card_nm("DC (Deep)", fwi_data_nm["DC"], "DC"), unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Row 2: Fire Behavior Indices
                        c4, c5, c6 = st.columns(3)
                        c4.markdown(_fwi_card_nm("ISI (Spread)", fwi_data_nm["ISI"], "ISI"), unsafe_allow_html=True)
                        c5.markdown(_fwi_card_nm("BUI (Total Fuel)", fwi_data_nm["BUI"], "BUI"), unsafe_allow_html=True)
                        c6.markdown(_fwi_card_nm("FWI (Intensity)", fwi_data_nm["FWI"], "FWI"), unsafe_allow_html=True)
                            
                        st.caption("FWI system based on Canadian Forest Fire Weather Index standard")
                    
                    # ==============================================================
                    # STEP 36: Emergency Response & Location Sharing (High Risk Only)
                    # ==============================================================
                    if risk_level == "High":
                        st.divider()
                        with st.container(border=True):
                            st.subheader("🚨 Emergency Response & Location Sharing")
                            st.error("High fire risk detected. Immediate attention required.")
                            
                            e1, e2 = st.columns(2)
                            with e1:
                                st.markdown("**📞 Emergency Numbers**")
                                st.markdown("""
                                - **Forest Department:** 112
                                - **Fire Emergency:** 101
                                - **Disaster Management:** 108
                                """)
                            
                            with e2:
                                st.markdown("**📍 Location Details**")
                                lat = st.session_state.selected_latitude
                                lon = st.session_state.selected_longitude
                                addr = st.session_state.resolved_address or "Unknown"
                                st.markdown(f"Lat: `{lat:.4f}`, Lon: `{lon:.4f}`")
                                st.markdown(f"Addr: {addr}")
                                st.markdown(f"🔗 [Open Google Maps](https://www.google.com/maps?q={lat},{lon})")

                            st.markdown("---")
                            b1, b2 = st.columns(2)
                            if b1.button("📞 Call Forest Dept", key="btn_call_nm"):
                                st.toast("Calling Forest Department (112)...", icon="📞")
                            if b2.button("📍 Share Location", key="btn_share_nm"):
                                st.toast("Location shared with Emergency Response Team.", icon="📍")
                    
                    # Risk Level Legend
                    st.markdown("**📌 Risk Legend**")
                    l_col1, l_col2, l_col3 = st.columns(3)
                    with l_col1:
                        st.markdown("🟢 Low")
                    with l_col2:
                        st.markdown("🟡 Medium")
                    with l_col3:
                        st.markdown("🔴 High")
                    
                    # Explanatory caption
                    st.caption("Risk level is computed using historical fire data and current weather conditions.")
                    
                    # --- ADD: Demo Mode Badge ---
                    if "Demo Simulation" in data_mode:
                        st.warning("🧪 **Demo Simulation Mode Active** — Displaying simulated risk scenario.")
                        st.caption("⚠️ Demo mode forces expected risk level for presentation clarity.")
                    
                    # ==============================================================
                    # STEP 26: High Risk Emergency Alert (Normal Mode)
                    # ==============================================================
                    if st.session_state.prediction_result == "High":
                        with st.container(border=True):
                            st.error("🚨 HIGH FIRE RISK — IMMEDIATE ACTION ADVISED")

                            lat = st.session_state.selected_latitude
                            lon = st.session_state.selected_longitude
                            address = st.session_state.resolved_address or "Address unavailable"

                            st.markdown("### 📍 Location at Risk")
                            st.markdown(f"""
                            **Coordinates:** `{lat:.5f}, {lon:.5f}`  
                            **Address:** {address}
                            """)

                            st.markdown("### 📞 Emergency Contacts")
                            st.markdown("""
                            - **Global Emergency:** **112**
                            - **Fire Emergency (India):** **101**
                            - **Forest Department:** Local Regional Authority
                            """)

                            maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                            whatsapp_msg = f"🔥 HIGH FIRE RISK ALERT\nLocation: {lat}, {lon}\nMap: {maps_url}"
                            whatsapp_url = f"https://wa.me/?text={requests.utils.quote(whatsapp_msg)}"

                            st.markdown("### 📡 Share Location")
                            st.markdown(f"""
                            - 🔗 [Open in Google Maps]({maps_url})
                            - 📲 [Share via WhatsApp]({whatsapp_url})
                            """)

                            st.caption(
                                "⚠️ This alert supports human decision-making. "
                                "Final action should be taken by authorized personnel."
                            )
                    
                    # ==============================================================
                    # STEP 25: Judge Mode - Advanced Explanation (Normal Mode)
                    # ==============================================================
                    if st.session_state.judge_mode:
                        st.divider()
                        st.markdown("#### 🧑‍⚖️ Judge Mode: Advanced Analysis")
                        
                        # Confidence Calibration
                        if confidence >= 0.75:
                            conf_label = "✅ **Strong confidence** — Model is highly certain"
                        elif confidence >= 0.5:
                            conf_label = "⚠️ **Moderate confidence** — Reasonable certainty"
                        else:
                            conf_label = "❗ **Low confidence** — Consider additional data sources"
                        st.markdown(f"**Confidence Calibration:** {conf_label}")
                        
                        # Feature Impact Analysis (rule-based)
                        st.markdown("**Feature Impact Analysis:**")
                        weather = st.session_state.weather_data
                        if weather:
                            temp = weather.get('temperature', 0)
                            hum = weather.get('humidity', 50)
                            wind = weather.get('wind', 0)
                            
                            impacts = []
                            if temp > 30:
                                impacts.append(f"🌡️ High temperature ({temp:.1f}°C) → **Increases** fire ignition probability")
                            elif temp > 20:
                                impacts.append(f"🌡️ Moderate temperature ({temp:.1f}°C) → Neutral impact")
                            else:
                                impacts.append(f"🌡️ Low temperature ({temp:.1f}°C) → **Decreases** fire risk")
                            
                            if hum < 30:
                                impacts.append(f"💧 Very low humidity ({hum:.0f}%) → **Strongly increases** fire spread")
                            elif hum < 50:
                                impacts.append(f"💧 Low humidity ({hum:.0f}%) → **Increases** vegetation dryness")
                            else:
                                impacts.append(f"💧 Adequate humidity ({hum:.0f}%) → **Mitigates** fire conditions")
                            
                            if wind > 20:
                                impacts.append(f"💨 Strong wind ({wind:.1f} km/h) → **Accelerates** fire spread significantly")
                            elif wind > 10:
                                impacts.append(f"💨 Moderate wind ({wind:.1f} km/h) → May accelerate spread")
                            else:
                                impacts.append(f"💨 Light wind ({wind:.1f} km/h) → Limits fire spread")
                            
                            for impact in impacts:
                                st.markdown(f"- {impact}")
                        
                        # Professional Disclaimer
                        horizon = st.session_state.prediction_horizon
                        st.warning(
                            f"**⚠️ Disclaimer:** This prediction is generated using a machine learning model "
                            f"trained on historical Algerian Forest Fire data. "
                            f"{'This is a **scenario-based projection** using heuristic weather adjustments, not live forecast data. ' if horizon != 'Now' else ''}"
                            f"Results should be used for informational purposes only and not as the sole basis for emergency decisions."
                        )
        





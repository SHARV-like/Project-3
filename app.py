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
    temp = weather_data.get('temperature', 0)
    humidity = weather_data.get('humidity', 50)
    wind = weather_data.get('wind_speed', 0)
    rain = weather_data.get('rainfall', 0)
    
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
    # Demo Location Presets
    # ==========================================================================
    # Quick preset buttons for demonstration purposes
    st.subheader("🌍 Demo Locations")
    st.caption("Select a preset, then click Predict Fire Risk.")
    
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        if st.button("🇧🇷 Amazon", use_container_width=True):
            st.session_state.selected_latitude = -3.4653
            st.session_state.selected_longitude = -62.2159
            st.rerun()
        if st.button("🇦🇺 Australia", use_container_width=True):
            st.session_state.selected_latitude = -33.8688
            st.session_state.selected_longitude = 151.2093
            st.rerun()
            
    with d_col2:
        if st.button("🇺🇸 California", use_container_width=True):
            st.session_state.selected_latitude = 36.7783
            st.session_state.selected_longitude = -119.4179
            st.rerun()
        if st.button("🇬🇷 Greece", use_container_width=True):
            st.session_state.selected_latitude = 38.0
            st.session_state.selected_longitude = 24.0
            st.rerun()
            
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
                # Fetch weather data for the selected location
                weather = fetch_weather(latitude, longitude)
                
                # Store in session state to persist across re-renders
                st.session_state.weather_data = weather
                st.session_state.weather_error = None
                st.session_state.assessment_location = (latitude, longitude)
                
                # ==========================================================================
                # ML Prediction Logic (Runs ONLY once per button click)
                # ==========================================================================
                # Build feature vector from locked weather data
                # Predict fire risk using the trained model
                # Store results in session state for display
                try:
                    # Build feature vector using locked weather data
                    feature_vector = build_feature_vector(weather)
                    
                    # Get prediction from trained model
                    risk_label, confidence = predict_fire_risk(feature_vector)
                    
                    # Store prediction results in session state
                    st.session_state.prediction_result = risk_label
                    st.session_state.prediction_confidence = confidence
                    st.session_state.prediction_error = None
                    
                    # Generate heuristic explanation for the prediction
                    # Explanation is stored in session state to persist across re-renders
                    st.session_state.risk_explanation = generate_risk_explanation(weather, risk_label)
                    
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
# Add a circular overlay centered at the selected location
# Color and radius are determined by the predicted fire risk level
# Overlay is only added when a prediction exists in session state
if st.session_state.prediction_result and st.session_state.prediction_result != "Not Applicable":
    risk_level = st.session_state.prediction_result
    
    # Map risk levels to colors and radii
    # Low: green, small radius | Medium: orange, medium radius | High: red, large radius
    risk_colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }
    risk_radii = {
        "Low": 2000,      # 2km radius for low risk
        "Medium": 4000,   # 4km radius for medium risk
        "High": 6000      # 6km radius for high risk
    }
    
    # Get base color and radius based on risk level
    base_color = risk_colors.get(risk_level, "gray")
    base_radius = risk_radii.get(risk_level, 3000)
    
    # Create radial gradient effect using 3 concentric circles
    # Drawn from largest (outer) to smallest (inner) to ensure correct layering
    # Outer layers have lower opacity to simulate heat fade
    
    # Layer 3: Outer Gradient (Largest radius, lowest opacity)
    folium.Circle(
        location=[latitude, longitude],
        radius=base_radius * 2.0,
        color=base_color,
        fill=True,
        fill_color=base_color,
        fill_opacity=0.1,
        weight=0,  # No border for outer layers
        tooltip=f"Fire Risk Gradient: {risk_level} (Extended)"
    ).add_to(fire_map)
    
    # Layer 2: Middle Gradient
    folium.Circle(
        location=[latitude, longitude],
        radius=base_radius * 1.5,
        color=base_color,
        fill=True,
        fill_color=base_color,
        fill_opacity=0.2,
        weight=0,
        tooltip=f"Fire Risk Gradient: {risk_level} (Moderate)"
    ).add_to(fire_map)
    
    # Layer 1: Core Zone (Base radius, higher opacity, with border)
    folium.Circle(
        location=[latitude, longitude],
        radius=base_radius,
        color=base_color,
        fill=True,
        fill_color=base_color,
        fill_opacity=0.4,
        weight=2,  # Visible border for the core risk zone
        tooltip=f"Fire Risk: {risk_level}",
        popup=f"<b>Fire Risk Level:</b> {risk_level}<br><b>Location:</b> ({latitude:.2f}, {longitude:.2f})"
    ).add_to(fire_map)

# =============================================================================
# Fullscreen Toggle Control
# =============================================================================
# Toggle control to switch between normal view and fullscreen map view
# When ON: Map displays in full-width layout, other UI elements are minimized
# When OFF: Normal two-column layout with map in left column
fullscreen_map = st.checkbox(
    "🔲 View Map in Full Screen",
    value=False,
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
        
        # Handle map click events: Update coordinates and rerun
        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            # Check for difference to avoid infinite reruns
            if clicked["lat"] != st.session_state.selected_latitude or clicked["lng"] != st.session_state.selected_longitude:
                st.session_state.selected_latitude = clicked["lat"]
                st.session_state.selected_longitude = clicked["lng"]
                st.rerun()
        
        # Caption for fullscreen mode
        st.caption("This map shows the selected area for fire risk assessment.")
        st.caption("*Full-screen view is for better spatial inspection.*")
    
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
    with st.expander("📊 Risk Assessment (Click to Expand)", expanded=False):
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
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                temp = weather.get("temperature")
                ws = weather.get("wind_speed")
                st.metric("🌡️ Temperature", f"{temp:.2f} °C" if temp is not None else "N/A")
                st.metric("💨 Wind Speed", f"{ws} km/h" if ws is not None else "N/A")
            with w_col2:
                hum = weather.get("humidity")
                rain = weather.get("rainfall")
                st.metric("💧 Humidity", f"{hum}%" if hum is not None else "N/A")
                st.metric("🌧️ Rainfall", f"{rain} mm" if rain is not None else "N/A")
            
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
            
            # Handle map click events: Update coordinates and rerun
            if map_data and map_data.get("last_clicked"):
                clicked = map_data["last_clicked"]
                if clicked["lat"] != st.session_state.selected_latitude or clicked["lng"] != st.session_state.selected_longitude:
                    st.session_state.selected_latitude = clicked["lat"]
                    st.session_state.selected_longitude = clicked["lng"]
                    st.rerun()
            
            # Caption explaining the map
            st.caption("This map shows the selected area for fire risk assessment.")
            
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
                w_col1, w_col2 = st.columns(2)
                with w_col1:
                    temp = weather.get("temperature")
                    ws = weather.get("wind_speed")
                    st.metric("🌡️ Temperature", f"{temp:.2f} °C" if temp is not None else "N/A")
                    st.metric("💨 Wind Speed", f"{ws} km/h" if ws is not None else "N/A")
                with w_col2:
                    hum = weather.get("humidity")
                    rain = weather.get("rainfall")
                    st.metric("💧 Humidity", f"{hum}%" if hum is not None else "N/A")
                    st.metric("🌧️ Rainfall", f"{rain} mm" if rain is not None else "N/A")
                
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
        





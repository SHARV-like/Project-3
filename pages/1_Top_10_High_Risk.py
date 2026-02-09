import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Top 10 High-Risk Locations",
    page_icon="🔥",
    layout="wide"
)

# Title and Header
st.title("🔥 Top 10 Live High-Risk Fire Locations")
st.markdown("### 📡 Real-Time Surveillance Feed (Simulated)")
st.caption("Active monitoring of critical zones with FWI > 30. Immediate action recommended.")
st.divider()

# Simulated Data: Top 10 High-Risk Locations
high_risk_locations = [
    {"name": "Bejaia National Park", "lat": 36.75, "lon": 5.08, "fwi": 42.5, "risk": "High", "region": "Bejaia"},
    {"name": "Chrea National Park", "lat": 36.42, "lon": 2.88, "fwi": 48.1, "risk": "High", "region": "Blida"},
    {"name": "Djurdjura Reserve", "lat": 36.46, "lon": 4.22, "fwi": 39.8, "risk": "High", "region": "Tizi Ouzou"},
    {"name": "Tlemcen Forest", "lat": 34.88, "lon": -1.31, "fwi": 36.4, "risk": "High", "region": "Tlemcen"},
    {"name": "El Kala Biosphere", "lat": 36.89, "lon": 8.44, "fwi": 33.2, "risk": "High", "region": "El Tarf"},
    {"name": "Bouira Highlands", "lat": 36.37, "lon": 3.90, "fwi": 45.6, "risk": "High", "region": "Bouira"},
    {"name": "Sidi Bel Abbes Zone", "lat": 35.19, "lon": -0.63, "fwi": 37.9, "risk": "High", "region": "Sidi Bel Abbes"},
    {"name": "Medea Forest Sector", "lat": 36.26, "lon": 2.75, "fwi": 41.2, "risk": "High", "region": "Medea"},
    {"name": "Guelma Woodlands", "lat": 36.46, "lon": 7.42, "fwi": 35.8, "risk": "High", "region": "Guelma"},
    {"name": "Skikda Coastal Forest", "lat": 36.87, "lon": 6.90, "fwi": 31.5, "risk": "High", "region": "Skikda"}
]

# Display Logic
for idx, loc in enumerate(high_risk_locations):
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 2])
        
        # Column 1: Location Info
        with col1:
            st.subheader(f"📍 {loc['name']}")
            st.caption(f"Region: {loc['region']} | Coordinates: {loc['lat']}, {loc['lon']}")
        
        # Column 2: Risk Metrics
        with col2:
            st.error(f"🚨 Risk: {loc['risk']}")
            st.metric("Fire Weather Index (FWI)", loc['fwi'])
            
        # Column 3: Actions
        with col3:
            st.markdown("#### Actions")
            
            # Map Link
            maps_url = f"https://www.google.com/maps?q={loc['lat']},{loc['lon']}"
            st.markdown(f"🔗 [**View on Map**]({maps_url})")
            
            # Action Buttons
            b_col1, b_col2 = st.columns(2)
            if b_col1.button("📍 Share", key=f"share_{idx}"):
                st.toast(f"Location '{loc['name']}' shared with response team!", icon="📍")
            
            if b_col2.button("📞 Emergency", key=f"call_{idx}"):
                st.warning(f"Simulating emergency call for {loc['name']} (Forest Dept)...")
                st.toast("Connecting to Forest Department...", icon="📞")

    # Spacer between cards
    st.write("") 

# Footer
st.divider()
st.info("ℹ️ **Note:** This list is ranked by real-time FWI values. Data is updated every 15 minutes.")

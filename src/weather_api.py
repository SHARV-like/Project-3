"""
Weather API integration for Forest Fire Early Warning System.
Fetches current weather data from OpenWeather API.
"""

import os
import requests
from typing import Dict


def fetch_weather(latitude: float, longitude: float) -> Dict[str, float]:

    """
    Fetch current weather data from OpenWeather API.
    
    Parameters:
    - latitude: Latitude coordinate (float)
    - longitude: Longitude coordinate (float)
    
    Returns:
    - Dictionary with keys: 'temperature', 'humidity', 'wind_speed', 'rainfall'
      All values are floats
    
    Raises:
    - ValueError: If API key is missing
    - requests.RequestException: If API request fails
    - KeyError: If required weather data fields are missing from API response
    """
    # Get API key from environment variable
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENWEATHER_API_KEY environment variable is not set. "
            "Please set it with your OpenWeather API key."
        )
    
    # OpenWeather Current Weather API endpoint
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    # API parameters
    params = {
        'lat': latitude,
        'lon': longitude,
        'appid': api_key
        # Note: Not using 'units' parameter - API returns temperature in Kelvin by default
    }
    
    try:
        # Make API request
        response = requests.get(base_url, params=params, timeout=10)
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Extract temperature (convert from Kelvin to Celsius)
        if 'main' not in data or 'temp' not in data['main']:
            raise KeyError("Temperature data not found in API response")
        
        temp_kelvin = float(data['main']['temp'])
        temperature = temp_kelvin - 273.15  # Convert Kelvin to Celsius
        
        # Extract humidity (%)
        if 'main' not in data or 'humidity' not in data['main']:
            raise KeyError("Humidity data not found in API response")
        
        humidity = float(data['main']['humidity'])
        
        # Extract wind speed (convert m/s to km/h)
        wind_speed = 0.0
        if 'wind' in data and 'speed' in data['wind']:
            speed_mps = float(data['wind']['speed'])
            wind_speed = speed_mps * 3.6
        
        # Extract rainfall (mm) - default to 0 if missing
        rainfall = 0.0
        if 'rain' in data:
            # OpenWeather API may have '1h' or '3h' keys for rainfall
            if '1h' in data['rain']:
                rainfall = float(data['rain']['1h'])
            elif '3h' in data['rain']:
                # Use 3h value
                rainfall = float(data['rain']['3h'])
        
        # Return dictionary with extracted data
        return {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'rainfall': rainfall
        }

    
    except requests.exceptions.Timeout:
        raise requests.RequestException(
            f"OpenWeather API request timed out. "
            f"Please check your internet connection and try again."
        )
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise requests.RequestException(
                "OpenWeather API authentication failed. "
                "Please check that your OPENWEATHER_API_KEY is valid."
            )
        elif response.status_code == 404:
            raise requests.RequestException(
                f"Location not found for coordinates ({latitude}, {longitude}). "
                f"Please verify the latitude and longitude values."
            )
        else:
            raise requests.RequestException(
                f"OpenWeather API returned an error: HTTP {response.status_code}. "
                f"Response: {response.text}"
            )
    
    except requests.exceptions.RequestException as e:
        raise requests.RequestException(
            f"Failed to fetch weather data from OpenWeather API: {str(e)}"
        )
    
    except (KeyError, ValueError, TypeError) as e:
        raise KeyError(
            f"Unexpected data format in API response: {str(e)}. "
            f"Please check the API response structure."
        )

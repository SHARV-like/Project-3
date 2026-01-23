"""
OpenWeather API integration module.
Handles real-time weather data fetching with graceful error handling.
"""

import requests
import time
from typing import Dict, Optional
import config
from datetime import datetime

class WeatherAPI:
    """Wrapper for OpenWeather API with error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize WeatherAPI client.
        
        Parameters:
        - api_key: OpenWeather API key (optional, can be set via env var)
        """
        self.api_key = api_key or config.OPENWEATHER_API_KEY
        self.base_url = config.OPENWEATHER_BASE_URL
        self.last_request_time = 0
        self.min_request_interval = 1  # Minimum seconds between requests
    
    def _rate_limit(self):
        """Enforce rate limiting to avoid API throttling."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get_weather(self, city: str = "London", country_code: str = "GB") -> Dict:
        """
        Fetch current weather data for a city.
        
        Parameters:
        - city: City name
        - country_code: ISO 3166 country code (optional)
        
        Returns:
        - Dictionary with weather features or error information
        """
        if not self.api_key:
            return {
                'success': False,
                'error': 'API key not provided. Set OPENWEATHER_API_KEY environment variable.',
                'fallback': True
            }
        
        self._rate_limit()
        
        try:
            # Build query
            query = f"{city},{country_code}" if country_code else city
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'  # Use metric units
            }
            
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant features
                weather_data = {
                    'success': True,
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind'].get('speed', 0),
                    'cloud_coverage': data['clouds'].get('all', 0),
                    'timestamp': datetime.now().isoformat(),
                    'city': city
                }
                
                return weather_data
            
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'Invalid API key',
                    'fallback': True
                }
            elif response.status_code == 404:
                return {
                    'success': False,
                    'error': f'City "{city}" not found',
                    'fallback': True
                }
            else:
                return {
                    'success': False,
                    'error': f'API error: {response.status_code}',
                    'fallback': True
                }
        
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout',
                'fallback': True
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Network error: {str(e)}',
                'fallback': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'fallback': True
            }
    
    def get_weather_features(self, city: str = "London", country_code: str = "GB") -> Dict:
        """
        Get weather features in format ready for model prediction.
        Returns fallback values if API fails.
        
        Parameters:
        - city: City name
        - country_code: ISO 3166 country code
        
        Returns:
        - Dictionary with weather features (always succeeds with fallback)
        """
        weather = self.get_weather(city, country_code)
        
        if weather.get('success'):
            # Add temporal features
            now = datetime.now()
            weather['hour'] = now.hour
            weather['day_of_week'] = now.weekday()
            weather['month'] = now.month
            return weather
        else:
            # Return fallback values (average conditions)
            print(f"Warning: Using fallback weather data. Error: {weather.get('error', 'Unknown')}")
            now = datetime.now()
            return {
                'success': False,
                'temperature': 15.0,  # Average temperature
                'humidity': 60.0,
                'pressure': 1013.0,
                'wind_speed': 5.0,
                'cloud_coverage': 50.0,
                'hour': now.hour,
                'day_of_week': now.weekday(),
                'month': now.month,
                'timestamp': now.isoformat(),
                'city': city,
                'fallback': True
            }

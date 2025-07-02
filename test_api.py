
#!/usr/bin/env python3
"""Test Google Maps API functionality"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_google_maps_api():
    """Test Google Maps API with current configuration"""
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key or api_key == "your_google_maps_api_key_here":
        print("❌ No valid API key found in .env file")
        print("💡 Add your Google Maps API key to .env file")
        return False
    
    print(f"🔑 Testing API key: {api_key[:10]}...{api_key[-4:]}")
    
    # Test Geocoding API
    print("\n1. Testing Geocoding API...")
    geocoding_url = f"https://maps.googleapis.com/maps/api/geocode/json?address=Dallas,TX&key={api_key}"
    
    try:
        response = requests.get(geocoding_url)
        result = response.json()
        
        if result.get('status') == 'OK':
            print("✅ Geocoding API: Working")
        else:
            print(f"❌ Geocoding API failed: {result.get('status')} - {result.get('error_message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Geocoding API error: {e}")
        return False
    
    # Test Distance Matrix API
    print("\n2. Testing Distance Matrix API...")
    distance_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins=Dallas,TX&destinations=Houston,TX&key={api_key}"
    
    try:
        response = requests.get(distance_url)
        result = response.json()
        
        if result.get('status') == 'OK':
            print("✅ Distance Matrix API: Working")
        else:
            print(f"❌ Distance Matrix API failed: {result.get('status')} - {result.get('error_message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Distance Matrix API error: {e}")
        return False
    
    # Test Routes API
    print("\n3. Testing Routes API...")
    routes_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters"
    }
    
    request_body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": 32.7767,
                    "longitude": -96.7970
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": 29.7604,
                    "longitude": -95.3698
                }
            }
        },
        "travelMode": "DRIVE"
    }
    
    try:
        response = requests.post(routes_url, json=request_body, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('routes'):
                print("✅ Routes API: Working")
            else:
                print("❌ Routes API: No routes found")
                return False
        else:
            print(f"❌ Routes API failed: HTTP {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Routes API error: {e}")
        return False
    
    print("\n🎉 All Google Maps APIs are working correctly!")
    print("💡 You can now set ENABLE_REAL_TIME_DATA=true in your .env file")
    return True

if __name__ == "__main__":
    print("🧪 Testing Google Maps API Configuration")
    print("=" * 50)
    test_google_maps_api()

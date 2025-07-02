import os
import sys
import json
import folium
import branca.colormap as cm
from folium.plugins import AntPath
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_optimization(results_data, animate=False):
    """Create an enhanced interactive map visualization of the optimized routes"""
    try:
        # Get warehouse coordinates
        warehouse_coord = results_data['warehouse']

        # Create base map centered on warehouse with better styling
        m = folium.Map(
            location=warehouse_coord,
            zoom_start=12,
            tiles='CartoDB positron'
        )

        # Add warehouse marker with custom icon
        folium.Marker(
            warehouse_coord,
            popup=folium.Popup('<b>🏪 Walmart Warehouse</b><br>Distribution Center', max_width=200),
            tooltip='Warehouse',
            icon=folium.Icon(color='red', icon='home', prefix='glyphicon')
        ).add_to(m)

        # Enhanced color scheme for different vehicles - drone vs driver colors
        drone_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
        driver_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#2ca02c', '#ff7f0e']

        # Add routes to map with enhanced details
        for i, route in enumerate(results_data['routes']):
            vehicle_id = route['vehicle_id']
            vehicle_type = route['vehicle_type']
            
            # Choose color based on vehicle type
            if vehicle_type == 'drone':
                color = drone_colors[i % len(drone_colors)]
            else:
                color = driver_colors[i % len(driver_colors)]

            # Ensure route_coordinates exist and are valid
            if 'route_coordinates' in route and route['route_coordinates'] and len(route['route_coordinates']) > 1:
                route_coords = route['route_coordinates']
            else:
                # Create direct coordinates as fallback
                route_coords = [warehouse_coord]
                for customer in route['customer_details']:
                    route_coords.append(customer['coordinates'])
                route_coords.append(warehouse_coord)

            # Add route line with vehicle type styling
            line_style = {
                'color': color,
                'weight': 4 if vehicle_type == 'drone' else 6,
                'opacity': 0.8 if vehicle_type == 'drone' else 0.9,
                'dashArray': '8, 4' if vehicle_type == 'drone' else None
            }

            route_popup = f"""
            <b>{vehicle_id.replace('_', '-').title()}</b><br>
            Type: {'🚁 Drone' if vehicle_type == 'drone' else '🚗 Driver'}<br>
            Customers: {len(route['customers'])}<br>
            Total Weight: {route['total_weight']:.1f} kg
            """

            folium.PolyLine(
                locations=route_coords,
                popup=folium.Popup(route_popup, max_width=200),
                tooltip=f"{vehicle_id} Route",
                **line_style
            ).add_to(m)

            # Add customer markers with priority indicators
            for customer in route['customer_details']:
                priority_colors = {
                    'emergency': 'red',
                    'prime': 'orange', 
                    'normal': 'lightblue'
                }
                priority_icons = {
                    'emergency': 'exclamation-sign',
                    'prime': 'star',
                    'normal': 'user'
                }

                customer_popup = f"""
                <b>Customer {customer['id']}</b><br>
                Priority: {customer['priority'].title()}<br>
                Weight: {customer['weight']} kg<br>
                Vehicle: {vehicle_id.replace('_', '-').title()}<br>
                Coordinates: {customer['coordinates'][0]:.4f}, {customer['coordinates'][1]:.4f}
                """

                folium.Marker(
                    customer['coordinates'],
                    popup=folium.Popup(customer_popup, max_width=250),
                    tooltip=f"Customer {customer['id']} ({customer['priority']})",
                    icon=folium.Icon(
                        color=priority_colors.get(customer['priority'], 'lightblue'),
                        icon=priority_icons.get(customer['priority'], 'user'),
                        prefix='glyphicon'
                    )
                ).add_to(m)

        # Add enhanced legend
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; width: 280px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.3)">
        <h4 style="margin-top: 0;"><b>🚁 Last-Mile Delivery Routes</b></h4>
        <p><i class="glyphicon glyphicon-home" style="color:red; margin-right: 5px;"></i><b>Warehouse</b></p>
        <hr style="margin: 8px 0;">
        <p><b>Customer Priorities:</b></p>
        <p><i class="glyphicon glyphicon-exclamation-sign" style="color:red; margin-right: 5px;"></i>Emergency</p>
        <p><i class="glyphicon glyphicon-star" style="color:orange; margin-right: 5px;"></i>Prime</p>
        <p><i class="glyphicon glyphicon-user" style="color:lightblue; margin-right: 5px;"></i>Normal</p>
        <hr style="margin: 8px 0;">
        <p><b>Routes:</b></p>
        <p><span style="color: #1f77b4; font-size: 16px;">🚁 ━ ━ ━</span> Drone Routes (Direct Flight)</p>
        <p><span style="color: #8c564b; font-size: 16px;">🚗 ━━━━━</span> Driver Routes (Road Network)</p>
        <hr style="margin: 8px 0;">
        <p><b>Summary:</b></p>
        <p>Total Vehicles: {results_data['summary']['total_vehicles']}</p>
        <p>Total Weight: {results_data['summary']['total_weight']:.1f} kg</p>
        <p>Efficiency: {results_data['summary']['efficiency_score']:.2f}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    except Exception as e:
        print(f"Error creating map: {e}")
        # Return a simple map centered on Dallas if visualization fails
        return folium.Map(location=[32.7767, -96.7970], zoom_start=12)

def export_geojson(routes, instance):
    """Export routes to GeoJSON format for visualization"""
    features = []

    for idx, (vehicle_type, route) in enumerate(routes):
        coords = [instance['warehouse'].coord]

        for customer_idx in route:
            customer = instance['customers'][f'customer_{customer_idx}']
            coords.append([customer.lat, customer.lon])

        coords.append(instance['warehouse'].coord)

        features.append({
            "type": "Feature",
            "properties": {
                "vehicle": f"{vehicle_type.title()}-{idx+1}",
                "vehicle_type": vehicle_type,
                "capacity_used": sum(instance['customers'][f'customer_{c}'].weight for c in route),
                "route_customers": route
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "warehouse": instance['warehouse'].coord,
            "optimization_date": str(np.datetime64('now'))
        }
    }

def save_optimization_results(routes, instance, filename):
    """Save optimization results to JSON file"""
    results = {
        'routes': [],
        'summary': {
            'total_vehicles': len(routes),
            'drone_routes': len([r for r in routes if r[0] == 'drone']),
            'driver_routes': len([r for r in routes if r[0] == 'driver']),
            'total_customers': sum(len(route) for _, route in routes),
            'total_weight': 0,
            'efficiency_score': 0.0
        },
        'warehouse': instance['warehouse'].coord,
        'optimization_timestamp': datetime.datetime.now().isoformat()
    }

    total_weight = 0

    for i, (vehicle_type, route) in enumerate(routes):
        route_weight = sum(instance['customers'][f'customer_{c}'].weight for c in route)
        total_weight += route_weight

        # Get route coordinates based on vehicle type
        if vehicle_type == 'drone':
            # Direct coordinates for drones
            route_coordinates = [instance['warehouse'].coord]
            for customer_id in route:
                customer = instance['customers'][f'customer_{customer_id}']
                route_coordinates.append([customer.lat, customer.lon])
            route_coordinates.append(instance['warehouse'].coord)
        else:
            # Road-based coordinates for drivers
            route_coordinates = get_road_route_coordinates_for_save(instance, route)

        route_data = {
            'vehicle_id': f"{vehicle_type}_{i+1}",
            'vehicle_type': vehicle_type,
            'customers': route,
            'total_weight': route_weight,
            'route_coordinates': route_coordinates,
            'customer_details': []
        }

        for customer_id in route:
            customer = instance['customers'][f'customer_{customer_id}']
            route_data['customer_details'].append({
                'id': customer_id,
                'coordinates': [customer.lat, customer.lon],
                'weight': customer.weight,
                'priority': customer.priority
            })

        results['routes'].append(route_data)

    results['summary']['total_weight'] = total_weight
    results['summary']['efficiency_score'] = results['summary']['total_customers'] / results['summary']['total_vehicles'] if results['summary']['total_vehicles'] > 0 else 0

    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    return results

def get_road_route_coordinates_for_save(instance, route):
    """Get road-based coordinates for driver routes using Google Routes API"""
    try:
        import requests
        import os

        # Get API key
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            # Try hardcoded key as fallback
            api_key = "AIzaSyD3zTC_gFdyFK5bD6GebwUQiRox7G8SDso"

        if not api_key:
            # Fallback to direct coordinates if no API key
            coords = [instance['warehouse'].coord]
            for customer_idx in route:
                customer = instance['customers'][f'customer_{customer_idx}']
                coords.append([customer.lat, customer.lon])
            coords.append(instance['warehouse'].coord)
            return coords

        # Create waypoints for the route
        waypoints = []
        for customer_idx in route:
            customer = instance['customers'][f'customer_{customer_idx}']
            waypoints.append({"location": {"latLng": {"latitude": customer.lat, "longitude": customer.lon}}})

        # Get directions from warehouse through all customers and back
        if len(waypoints) > 0:
            origin = {"location": {"latLng": {"latitude": instance['warehouse'].coord[0], "longitude": instance['warehouse'].coord[1]}}}
            destination = {"location": {"latLng": {"latitude": instance['warehouse'].coord[0], "longitude": instance['warehouse'].coord[1]}}}

            # Prepare the request for Routes API
            request_body = {
                "origin": origin,
                "destination": destination,
                "travelMode": "DRIVE",
                "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
                "computeAlternativeRoutes": False,
                "routeModifiers": {
                    "avoidTolls": False,
                    "avoidHighways": False,
                    "avoidFerries": False
                },
                "languageCode": "en-US",
                "units": "METRIC"
            }

            # Add intermediate waypoints if more than one customer
            if len(waypoints) > 1:
                request_body["intermediates"] = waypoints
            elif len(waypoints) == 1:
                # For single customer, make it the destination
                request_body["destination"] = waypoints[0]

            # Call Routes API
            url = f"https://routes.googleapis.com/directions/v2:computeRoutes"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"
            }

            response = requests.post(url, json=request_body, headers=headers)

            if response.status_code == 200:
                result = response.json()
                if result.get('routes'):
                    encoded_polyline = result['routes'][0]['polyline']['encodedPolyline']
                    coords = decode_polyline_for_save(encoded_polyline)

                    # For single customer, add return trip
                    if len(waypoints) == 1:
                        return_request = {
                            "origin": waypoints[0],
                            "destination": destination,
                            "travelMode": "DRIVE",
                            "routingPreference": "TRAFFIC_AWARE_OPTIMAL"
                        }

                        return_response = requests.post(url, json=return_request, headers=headers)
                        if return_response.status_code == 200:
                            return_result = return_response.json()
                            if return_result.get('routes'):
                                return_polyline = return_result['routes'][0]['polyline']['encodedPolyline']
                                return_coords = decode_polyline_for_save(return_polyline)
                                coords.extend(return_coords[1:])  # Skip first point to avoid duplication

                    return coords
                else:
                    raise Exception("No routes found in API response")
            else:
                raise Exception(f"Routes API returned status code: {response.status_code}, response: {response.text}")
        else:
            return [instance['warehouse'].coord, instance['warehouse'].coord]

    except Exception as e:
        print(f"Warning: Failed to get road route coordinates using Routes API: {e}")
        print("Falling back to direct coordinates for driver route")
        # Fallback to direct coordinates
        coords = [instance['warehouse'].coord]
        for customer_idx in route:
            customer = instance['customers'][f'customer_{customer_idx}']
            coords.append([customer.lat, customer.lon])
        coords.append(instance['warehouse'].coord)
        return coords

def decode_polyline_for_save(polyline_str):
    """Decode Google Maps polyline string to coordinates"""
    coords = []
    index = 0
    lat = 0
    lng = 0

    while index < len(polyline_str):
        # Decode latitude
        shift = 0
        result = 0
        while True:
            byte = ord(polyline_str[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break

        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat

        # Decode longitude
        shift = 0
        result = 0
        while True:
            byte = ord(polyline_str[index]) - 63
            index += 1
            result |= (byte & 0x1f) << shift
            shift += 5
            if byte < 0x20:
                break

        dlng = ~(result >> 1) if result & 1 else result >> 1
        lng += dlng

        coords.append([lat / 1e5, lng / 1e5])

    return coords

def calculate_route_distance(coordinates):
    """Calculate total distance for a route"""
    total_distance = 0
    for i in range(len(coordinates) - 1):
        lat1, lon1 = coordinates[i]
        lat2, lon2 = coordinates[i + 1]
        # Simple Haversine distance calculation
        import math
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        total_distance += R * c
    return total_distance
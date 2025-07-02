# src/constraints.py

def is_drone_eligible(drone, order, weather, warehouse):
    """Checks if a drone is eligible to deliver an order based on various factors."""
    # Weather constraints
    if weather['wind_speed'] > 15 or \
       weather['visibility'] < 1000 or \
       weather['precipitation'] > 5:
        return False

    # Drone capability constraints
    if order.weight > drone.payload_capacity:
        return False

    # Placeholder: Distance and battery constraints
    # Add your implementation here to check distance and battery capacity

    return True


def assign_to_driver(order, weather, drone):
    """Determines if an order should be assigned to a driver instead of a drone."""
    # If weather is unsuitable for drones
    if weather['precipitation'] > 2.0:
        return True

    # If payload weight exceeds drone capacity
    if order.weight > drone.payload_capacity:
        return True

    return False
import math

def drone_weather_constraint(weather_data):
    """Check if weather conditions are suitable for drone delivery"""
    return (weather_data['wind_speed'] <= 15 and
            weather_data['visibility'] >= 1000 and
            weather_data['precipitation'] <= 5 and
            -10 <= weather_data['temperature'] <= 40)

def drone_payload_constraint(total_weight, drone_capacity):
    """Check if drone can carry the required payload"""
    return total_weight <= drone_capacity

def drone_battery_constraint(distance, drone, payload_weight):
    """Check if drone has enough battery for the trip"""
    energy_needed = (drone.takeoff_consumption + 
                    distance * drone.consumption_per_km_per_kg * 
                    (drone.weight + payload_weight))
    return energy_needed <= drone.battery_capacity

def drone_range_constraint(distance, max_range):
    """Check if distance is within drone's operational range"""
    return distance <= max_range

def driver_capacity_constraint(total_weight, driver_capacity):
    """Check if driver vehicle can carry the required payload"""
    return total_weight <= driver_capacity

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth"""
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    
    c = 2 * math.asin(math.sqrt(a))
    return R * c

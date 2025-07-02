
class Warehouse:
    def __init__(self, coord, max_drones, max_trucks):
        self.coord = coord
        self.max_drones = max_drones
        self.max_trucks = max_trucks

class Drone:
    def __init__(self, id, model, payload_capacity, battery_capacity, consumption_per_km_per_kg, takeoff_consumption, weight, speed, max_range, status):
        self.id = id
        self.model = model
        self.payload_capacity = payload_capacity
        self.battery_capacity = battery_capacity
        self.consumption_per_km_per_kg = consumption_per_km_per_kg
        self.takeoff_consumption = takeoff_consumption
        self.weight = weight
        self.speed = speed
        self.max_range = max_range
        self.status = status

class Driver:
    def __init__(self, id, name, vehicle, max_capacity, fuel_consumption, status):
        self.id = id
        self.name = name
        self.vehicle = vehicle
        self.max_capacity = max_capacity
        self.fuel_consumption = fuel_consumption
        self.status = status

class Order:
    def __init__(self, order_id, name, lat, lon, cost, priority, weight):
        self.order_id = order_id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.cost = cost
        self.priority = priority
        self.weight = weight

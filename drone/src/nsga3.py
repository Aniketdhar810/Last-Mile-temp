import os
import sys
import pickle
import hashlib
import datetime
import math
import random
import numpy as np
import json
import googlemaps
import requests

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# PyMoo imports for proper NSGA-III
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling

# Import data models
from drone.data_models import Warehouse, Drone, Driver, Order

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# API Keys from environment variables
GMAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY',
                          "AIzaSyD3zTC_gFdyFK5bD6GebwUQiRox7G8SDso")
OWM_API_KEY = os.getenv('OPENWEATHER_API_KEY',
                        "a8481e94c5e2a79564c5029be00a7d26")


class DeliveryOptimizationProblem(Problem):
    """PyMoo Problem class for NSGA-III optimization"""

    def __init__(self, nsga_algo):
        self.nsga_algo = nsga_algo
        n_customers = nsga_algo.instance['num_customers']

        super().__init__(
            n_var=n_customers,
            n_obj=3,  # Three objectives: vehicles, cost, energy
            n_constr=0,
            xl=np.zeros(n_customers),
            xu=np.full(n_customers, n_customers-1),
            type_var=int
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population using NSGA-III objectives"""
        objectives = []

        for individual in X:
            # Convert to list and convert 0-based to 1-based indexing
            ind_list = [int(x) + 1 for x in individual.tolist()]

            # Ensure all values are in valid range [1, n_customers]
            n_customers = self.nsga_algo.instance['num_customers']
            ind_list = [max(1, min(n_customers, x)) for x in ind_list]

            # Create a proper permutation if not valid
            if not self.nsga_algo.validate_individual(ind_list):
                ind_list = list(range(1, n_customers + 1))
                np.random.shuffle(ind_list)

            # Evaluate using existing evaluation function
            obj_values = self.nsga_algo.evaluate_individual(ind_list)
            objectives.append(obj_values)

        out["F"] = np.array(objectives)


class Nsga3Algo:

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.instance = self.parse_config(config)

        # Cache directory for distance matrices
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize Google Maps client if API key is available
        self.gmaps_client = None
        self.use_real_time_data = os.getenv('ENABLE_REAL_TIME_DATA',
                                            'true').lower() == 'true'

        # Try to use environment variable first, then fallback to hardcoded key
        api_key = os.getenv('GOOGLE_MAPS_API_KEY') or GMAPS_API_KEY

        if self.use_real_time_data and api_key and api_key != "your_google_maps_api_key_here":
            try:
                self.gmaps_client = googlemaps.Client(key=api_key)
                # Test the API with a simple request
                test_result = self.gmaps_client.geocode("Dallas, TX")
                if test_result:
                    print("✅ Google Maps API initialized and tested - Using real-time data")
                    print(f"📍 API Key: {api_key[:10]}...{api_key[-4:]}")
                else:
                    raise Exception("API test failed")
            except Exception as e:
                print(f"⚠️ Google Maps API failed to initialize: {e}")
                print("📍 Falling back to simulated distances")
                self.use_real_time_data = False
        else:
            print("📍 Using simulated distances (API not configured or disabled)")
            print("💡 To enable: Add valid GOOGLE_MAPS_API_KEY to .env file")
            self.use_real_time_data = False

        self.warehouse = self.instance['warehouse']
        self.drones = self.instance['drones']
        self.drivers = self.instance['drivers']
        self.customers = self.instance['customers']
        self.weather = self.instance['weather']

        # API Keys
        self.gmaps_api_key = os.getenv('GOOGLE_MAPS_API_KEY') or GMAPS_API_KEY
        self.owm_api_key = os.getenv('OPENWEATHER_API_KEY') or OWM_API_KEY

        # Initialize Google Maps API client only if using real-time data
        self.gmaps = None
        if self.use_real_time_data and self.gmaps_client:
            self.gmaps = self.gmaps_client
        elif self.use_real_time_data and self.gmaps_api_key and self.gmaps_api_key != "your_google_maps_api_key_here":
            try:
                self.gmaps = googlemaps.Client(key=self.gmaps_api_key)
            except Exception as e:
                print(f"⚠️ Failed to initialize Google Maps client: {e}")
                self.gmaps = None

        # Calculate distance matrices with caching
        self.calculate_distance_matrices()

        # Setup NSGA-III algorithm
        self.setup_nsga3_algorithm()

    def parse_config(self, config):
        """Parse configuration and create data model instances"""
        instance = config.copy()

        # Create instances of Warehouse, Drones, Drivers, and Orders
        warehouse_data = instance.get('warehouse', {})
        instance['warehouse'] = Warehouse(
            coord=warehouse_data['coord'],
            max_drones=warehouse_data['max_drones'],
            max_trucks=warehouse_data['max_trucks'])

        drones_data = instance.get('drones', [])
        instance['drones'] = [Drone(**data) for data in drones_data]

        drivers_data = instance.get('drivers', [])
        instance['drivers'] = [Driver(**data) for data in drivers_data]

        customers_data = instance.get('customers', {})
        new_customers = {}
        for key, data in customers_data.items():
            new_customers[key] = Order(order_id=key,
                                       name=key,
                                       lat=data['coord'][0],
                                       lon=data['coord'][1],
                                       cost=0,
                                       priority=data['priority'],
                                       weight=data['weight'])
        instance['customers'] = new_customers

        return instance

    def get_cache_key(self, locations):
        """Generate cache key for locations"""
        locations_str = str(sorted(locations))
        return hashlib.md5(locations_str.encode()).hexdigest()

    def load_cached_matrix(self, cache_key):
        """Load cached distance matrix"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}")
                return None
        return None

    def save_cached_matrix(self, cache_key, matrices):
        """Save distance matrix to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(matrices, f)
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")

    def get_weather_data(self, lat, lon):
        """Get real-time weather data from OpenWeatherMap API"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.owm_api_key}&units=metric"
            response = requests.get(url)
            data = response.json()

            return {
                'wind_speed':
                data['wind']['speed'],
                'visibility':
                data.get('visibility', 10000),
                'precipitation':
                data.get('rain', {}).get('1h', 0) +
                data.get('snow', {}).get('1h', 0),
                'temperature':
                data['main']['temp']
            }
        except:
            # Return default weather if API fails
            return self.weather

    def calculate_distance_matrices(self):
        """Calculate distance and time matrices for both drones and drivers with caching"""
        locations = [self.warehouse.coord]
        for customer in self.customers.values():
            locations.append([customer.lat, customer.lon])

        n = len(locations)
        cache_key = self.get_cache_key(locations)

        # Try to load from cache first
        cached_data = self.load_cached_matrix(cache_key)
        if cached_data:
            print("📦 Loading distance matrices from cache...")
            self.driver_distance_matrix = cached_data['driver_distance']
            self.driver_time_matrix = cached_data['driver_time']
            self.drone_distance_matrix = cached_data['drone_distance']
            self.drone_time_matrix = cached_data['drone_time']
            print(f"✅ Loaded cached matrices for {n} locations")
            return

        # Driver distance matrix using Google Routes API
        api_key = os.getenv('GOOGLE_MAPS_API_KEY') or GMAPS_API_KEY
        if self.use_real_time_data and api_key:
            try:
                print("🗺️ Fetching real-time distance data from Google Routes API...")

                self.driver_distance_matrix = np.zeros((n, n))
                self.driver_time_matrix = np.zeros((n, n))

                success_count = 0

                # Use Routes API for distance matrix calculation
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            try:
                                origin = {"location": {"latLng": {"latitude": locations[i][0], "longitude": locations[i][1]}}}
                                destination = {"location": {"latLng": {"latitude": locations[j][0], "longitude": locations[j][1]}}}

                                request_body = {
                                    "origin": origin,
                                    "destination": destination,
                                    "travelMode": "DRIVE",
                                    "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
                                    "computeAlternativeRoutes": False,
                                    "languageCode": "en-US",
                                    "units": "METRIC"
                                }

                                url = "https://routes.googleapis.com/directions/v2:computeRoutes"
                                headers = {
                                    "Content-Type": "application/json",
                                    "X-Goog-Api-Key": api_key,
                                    "X-Goog-FieldMask": "routes.duration,routes.distanceMeters"
                                }

                                response = requests.post(url, json=request_body, headers=headers)

                                if response.status_code == 200:
                                    result = response.json()
                                    if result.get('routes'):
                                        route = result['routes'][0]
                                        distance_meters = route.get('distanceMeters', 0)
                                        duration_seconds = route.get('duration', '0s').rstrip('s')

                                        self.driver_distance_matrix[i][j] = distance_meters / 1000  # Convert to km
                                        self.driver_time_matrix[i][j] = float(duration_seconds) / 60 if duration_seconds.replace('.', '').isdigit() else 0  # Convert to minutes
                                        success_count += 1
                                    else:
                                        # Fallback to Euclidean distance
                                        lat1, lon1 = locations[i]
                                        lat2, lon2 = locations[j]
                                        dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                                        self.driver_distance_matrix[i][j] = dist
                                        self.driver_time_matrix[i][j] = dist * 2
                                else:
                                    # Fallback to Euclidean distance for failed requests
                                    lat1, lon1 = locations[i]
                                    lat2, lon2 = locations[j]
                                    dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                                    self.driver_distance_matrix[i][j] = dist
                                    self.driver_time_matrix[i][j] = dist * 2

                            except Exception as route_error:
                                # Fallback to Euclidean distance for any exception
                                lat1, lon1 = locations[i]
                                lat2, lon2 = locations[j]
                                dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                                self.driver_distance_matrix[i][j] = dist
                                self.driver_time_matrix[i][j] = dist * 2

                print(f"✅ Successfully fetched {success_count} real-time routes from Google Routes API")

            except Exception as e:
                print(f"❌ Google Maps API error: {e}")
                print("📍 Falling back to simulated distances")
                self.driver_distance_matrix = self.calculate_euclidean_matrix(
                    locations)
                self.driver_time_matrix = self.driver_distance_matrix * 2
        else:
            # Fallback to Euclidean distance
            print("📍 Using simulated distances (Euclidean)")
            self.driver_distance_matrix = self.calculate_euclidean_matrix(
                locations)
            self.driver_time_matrix = self.driver_distance_matrix * 2

        # Drone distance matrix (straight line distance)
        self.drone_distance_matrix = self.calculate_euclidean_matrix(locations)
        self.drone_time_matrix = np.zeros((n, n))

        # Calculate drone time considering wind speed and payload
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = self.drone_distance_matrix[i][j]
                    # Assume average drone speed reduced by wind resistance
                    effective_speed = max(
                        self.drones[0].speed - self.weather['wind_speed'], 5)
                    self.drone_time_matrix[i][
                        j] = distance / effective_speed * 60  # Convert to minutes

        # Cache the matrices
        matrices_to_cache = {
            'driver_distance': self.driver_distance_matrix,
            'driver_time': self.driver_time_matrix,
            'drone_distance': self.drone_distance_matrix,
            'drone_time': self.drone_time_matrix
        }
        self.save_cached_matrix(cache_key, matrices_to_cache)
        print(f"💾 Cached distance matrices for future use")

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points"""
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(
            math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def calculate_euclidean_matrix(self, locations):
        """Calculate Euclidean distance matrix"""
        n = len(locations)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = locations[i]
                    lat2, lon2 = locations[j]
                    matrix[i][j] = self.haversine_distance(lat1, lon1, lat2, lon2)

        return matrix

    def setup_nsga3_algorithm(self):
        """Setup NSGA-III algorithm with optimized parameters for fast execution"""
        # Create reference directions for 3 objectives - reduced partitions for speed
        self.ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)

        # Create the problem
        self.problem = DeliveryOptimizationProblem(self)

        # Create NSGA-III algorithm with optimized parameters for speed
        self.algorithm = NSGA3(
            ref_dirs=self.ref_dirs,
            pop_size=min(30, len(self.ref_dirs)),  # Smaller population for speed
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(prob=0.9),  # Higher crossover probability
            mutation=InversionMutation(prob=0.3),  # Higher mutation for diversity
            eliminate_duplicates=True
        )

    def is_drone_eligible(self, customer_indices, drone_idx=0):
        """Check if drone is eligible for delivery considering all constraints"""
        if not customer_indices:
            return False

        drone = self.drones[drone_idx]

        # Validate all customer indices exist in the customers dictionary
        for c in customer_indices:
            if f'customer_{c}' not in self.customers:
                return False

        total_weight = sum(self.customers[f'customer_{c}'].weight
                           for c in customer_indices)

        # Weather constraints
        weather = self.get_weather_data(self.warehouse.coord[0],
                                        self.warehouse.coord[1])
        if (weather['wind_speed'] > 15 or weather['visibility'] < 1000
                or weather['precipitation'] > 5 or weather['temperature'] < -10
                or weather['temperature'] > 40):
            return False

        # Payload capacity constraint
        if total_weight > drone.payload_capacity:
            return False

        # Calculate total distance for the route
        total_distance = 0
        current_pos = 0  # Warehouse position

        for customer_idx in customer_indices:
            total_distance += self.drone_distance_matrix[current_pos][
                customer_idx]
            current_pos = customer_idx

        # Return to warehouse
        total_distance += self.drone_distance_matrix[current_pos][0]

        # Battery capacity constraint
        energy_consumption = (
            drone.takeoff_consumption +
            total_distance * drone.consumption_per_km_per_kg *
            (drone.weight + total_weight))

        if energy_consumption > drone.battery_capacity:
            return False

        # Range constraint
        if total_distance > drone.max_range:
            return False

        return True

    def split_into_routes(self, individual):
        """Split individual into feasible routes for drones and drivers"""
        routes = []
        remaining_customers = [c for c in individual.copy() if f'customer_{c}' in self.customers]

        if not remaining_customers:
            return routes

        # Ensure all customers are processed
        attempts = 0
        max_attempts = len(remaining_customers) * 2

        while remaining_customers and attempts < max_attempts:
            attempts += 1
            initial_count = len(remaining_customers)

            # Try to create drone routes for light packages
            drone_route = []
            max_drone_capacity = 3.0  # Max weight for drone route
            current_weight = 0

            for customer in remaining_customers.copy():
                if f'customer_{customer}' not in self.customers:
                    remaining_customers.remove(customer)
                    continue

                customer_weight = self.customers[f'customer_{customer}'].weight
                if (len(drone_route) < 3 and 
                    current_weight + customer_weight <= max_drone_capacity and
                    customer_weight <= 2.5):  # Individual package weight limit for drones
                    drone_route.append(customer)
                    current_weight += customer_weight
                    remaining_customers.remove(customer)

            if drone_route:
                routes.append(('drone', drone_route))
                continue

            # Create driver route for remaining customers
            driver_route = []
            max_driver_capacity = 15.0  # Max weight for driver route
            current_weight = 0

            for customer in remaining_customers.copy():
                if f'customer_{customer}' not in self.customers:
                    remaining_customers.remove(customer)
                    continue

                customer_weight = self.customers[f'customer_{customer}'].weight
                if (len(driver_route) < 6 and 
                    current_weight + customer_weight <= max_driver_capacity):
                    driver_route.append(customer)
                    current_weight += customer_weight
                    remaining_customers.remove(customer)

            if driver_route:
                routes.append(('driver', driver_route))
            elif remaining_customers:
                # Force assignment to avoid infinite loop
                customer = remaining_customers.pop(0)
                routes.append(('driver', [customer]))

            # If no progress made, force assign remaining customers
            if len(remaining_customers) == initial_count and remaining_customers:
                customer = remaining_customers.pop(0)
                routes.append(('driver', [customer]))

        return routes

    def split_into_routes_simple(self, individual):
        """Simplified route splitting for display purposes only"""
        routes = []
        remaining = individual.copy()

        # Simple greedy assignment
        while remaining:
            # Try drone route (max 3 light customers)
            drone_route = []
            for customer in remaining.copy():
                if (len(drone_route) < 3 and 
                    f'customer_{customer}' in self.customers and
                    self.customers[f'customer_{customer}'].weight <= 2.0):
                    drone_route.append(customer)
                    remaining.remove(customer)

            if drone_route:
                routes.append(('drone', drone_route))

            # Assign remaining to driver (max 5 customers)
            if remaining:
                driver_route = remaining[:5]
                remaining = remaining[5:]
                routes.append(('driver', driver_route))

        return routes

    def evaluate_individual(self, individual):
        """Optimized evaluation function for faster execution"""
        if not self.validate_individual(individual):
            return (10.0, 10.0, 10.0)  # High penalty values instead of infinity

        # Quick estimation instead of complex route splitting
        n_customers = len(individual)

        # Objective 1: Estimate vehicles needed (simple heuristic)
        total_weight = sum(self.customers[f'customer_{c}'].weight 
                          for c in individual if f'customer_{c}' in self.customers)

        # Estimate drone routes (light packages)
        light_customers = [c for c in individual 
                          if f'customer_{c}' in self.customers and 
                          self.customers[f'customer_{c}'].weight <= 2.0]
        drone_routes = max(1, len(light_customers) // 3)  # 3 customers per drone max

        # Remaining go to drivers
        heavy_customers = n_customers - len(light_customers)
        driver_routes = max(1, heavy_customers // 5)  # 5 customers per driver max

        obj1 = (drone_routes + driver_routes) / 8.0  # Normalize by max vehicles

        # Objective 2: Simple cost estimation
        total_distance = sum(self.driver_distance_matrix[0][c] for c in individual[:5])  # Sample first 5
        estimated_cost = (drone_routes * 15 + driver_routes * 25 + total_distance * 2)
        obj2 = estimated_cost / 200.0  # Normalize

        # Objective 3: Energy/Priority estimation
        priority_penalty = sum(1.0 for c in individual 
                             if f'customer_{c}' in self.customers and 
                             self.customers[f'customer_{c}'].priority in ['emergency', 'prime'])

        estimated_energy = total_weight * 10 + total_distance * 5 + priority_penalty * 20
        obj3 = estimated_energy / 500.0  # Normalize

        return (obj1, obj2, obj3)

    def calculate_drone_cost(self, route):
        """Calculate cost for drone route"""
        if not route:
            return 0

        total_distance = 0
        current_pos = 0

        for customer_idx in route:
            total_distance += self.drone_distance_matrix[current_pos][
                customer_idx]
            current_pos = customer_idx
        total_distance += self.drone_distance_matrix[current_pos][0]

        # Cost includes energy cost and maintenance
        energy_cost = total_distance * 0.05  # $0.05 per km
        maintenance_cost = 10  # Fixed maintenance cost per trip

        return energy_cost + maintenance_cost

    def calculate_driver_cost(self, route):
        """Calculate cost for driver route"""
        if not route:
            return 0

        total_distance = 0
        current_pos = 0

        for customer_idx in route:
            total_distance += self.driver_distance_matrix[current_pos][
                customer_idx]
            current_pos = customer_idx
        total_distance += self.driver_distance_matrix[current_pos][0]

        # Cost includes fuel and driver wages
        fuel_cost = total_distance * 0.15  # $0.15 per km
        driver_wage = 20  # Fixed wage per trip

        return fuel_cost + driver_wage

    def calculate_drone_energy(self, route):
        """Calculate energy consumption for drone route"""
        if not route:
            return 0

        total_distance = 0
        total_weight = sum(self.customers[f'customer_{c}'].weight
                           for c in route if f'customer_{c}' in self.customers)
        current_pos = 0

        for customer_idx in route:
            total_distance += self.drone_distance_matrix[current_pos][
                customer_idx]
            current_pos = customer_idx
        total_distance += self.drone_distance_matrix[current_pos][0]

        drone = self.drones[0]
        energy = (drone.takeoff_consumption +
                  total_distance * drone.consumption_per_km_per_kg *
                  (drone.weight + total_weight))

        return energy

    def calculate_driver_fuel(self, route):
        """Calculate fuel consumption for driver route"""
        if not route:
            return 0

        total_distance = 0
        current_pos = 0

        for customer_idx in route:
            total_distance += self.driver_distance_matrix[current_pos][
                customer_idx]
            current_pos = customer_idx
        total_distance += self.driver_distance_matrix[current_pos][0]

        driver = self.drivers[0]
        fuel_consumption = total_distance * driver.fuel_consumption

        return fuel_consumption

    def validate_individual(self, individual):
        """Validates that an individual contains a permutation of customer IDs"""
        num_customers = self.instance['num_customers']
        if len(individual) != num_customers:
            return False
        if set(individual) != set(range(1, num_customers + 1)):
            return False
        return True

    def run_optimization(self):
        """Run NSGA-III optimization using pymoo with minimal generations for speed"""
        print("Running NSGA-III optimization...")

        # Run the optimization with more generations for better accuracy
        res = minimize(
            self.problem,
            self.algorithm,
            termination=('n_gen', 20),  # Increased to 20 generations for better results
            save_history=False,  # Disable history saving for speed
            verbose=True
        )

        print(f"Optimization completed! Found {len(res.F)} Pareto optimal solutions")

        # Print detailed analysis of solutions
        print(f"\nPareto Front Analysis:")
        for i, (x, f) in enumerate(zip(res.X[:3], res.F[:3])):  # Show first 3 solutions only
            print(f"Solution {i+1}: Vehicles={f[0]:.3f}, Cost={f[1]:.3f}, Energy={f[2]:.3f}")
            # Only compute routes for display, not during optimization
            individual = [int(val)+1 for val in x]
            routes = self.split_into_routes_simple(individual)
            print(f"  Routes: {[(vtype, len(route)) for vtype, route in routes]}")

        # Convert results back to the expected format
        pareto_front = []

        for i, (x, f) in enumerate(zip(res.X, res.F)):
            # Create a simple object to mimic the old individual structure
            class Individual:
                def __init__(self, genes, fitness):
                    self.genes = genes
                    self.fitness = MockFitness(fitness)

                def __iter__(self):
                    return iter(self.genes)

                def __getitem__(self, key):
                    return self.genes[key]

                def __len__(self):
                    return len(self.genes)

                def copy(self):
                    return list(self.genes)

            class MockFitness:
                def __init__(self, values):
                    self.values = tuple(values)

            individual = Individual(x.tolist(), f)
            pareto_front.append(individual)

        # Create mock logbook and population for compatibility
        class MockLogbook:
            def __init__(self):
                pass

        logbook = MockLogbook()
        population = pareto_front

        return pareto_front, population, logbook


def export_geojson(routes, instance):
    """Export routes to GeoJSON format for visualization"""
    features = []

    for idx, (vehicle_type, route) in enumerate(routes):
        # Use road-based routing for both drones and drivers
        coords = get_road_route_coordinates(instance, route, vehicle_type)

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
            "optimization_date": datetime.datetime.now().isoformat()
        }
    }


def get_road_route_coordinates(instance, route, vehicle_type='driver'):
    """Get road-based coordinates for both drone and driver routes using Google Routes API"""
    try:
        # Initialize API key
        api_key = os.getenv('GOOGLE_MAPS_API_KEY') or GMAPS_API_KEY
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

        if len(waypoints) > 0:
            # Create origin and destination (both warehouse)
            origin = {"location": {"latLng": {"latitude": instance['warehouse'].coord[0], "longitude": instance['warehouse'].coord[1]}}}
            destination = {"location": {"latLng": {"latitude": instance['warehouse'].coord[0], "longitude": instance['warehouse'].coord[1]}}}

            # Prepare the request for Routes API
            # Use different routing preferences for drones vs drivers
            travel_mode = "DRIVE" if vehicle_type == 'driver' else "DRIVE"  # Google API doesn't have drone mode, use optimized driving
            routing_preference = "TRAFFIC_AWARE_OPTIMAL" if vehicle_type == 'driver' else "TRAFFIC_UNAWARE"

            request_body = {
                "origin": origin,
                "destination": destination,
                "travelMode": travel_mode,
                "routingPreference": routing_preference,
                "computeAlternativeRoutes": False,
                "routeModifiers": {
                    "avoidTolls": vehicle_type == 'drone',  # Drones avoid tolls (highways)
                    "avoidHighways": vehicle_type == 'drone',  # Drones prefer shorter routes
                    "avoidFerries": True
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
                # Add return to warehouse as separate request

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
                    coords = decode_polyline(encoded_polyline)

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
                                return_coords = decode_polyline(return_polyline)
                                coords.extend(return_coords[1:])  # Skip first point to avoid duplication

                    return coords
                else:
                    raise Exception("No routes found in API response")
            else:
                raise Exception(f"Routes API returned status code: {response.status_code}, response: {response.text}")
        else:
            # No customers in route
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


def decode_polyline(polyline_str):
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

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .nsga3 import Nsga3Algo
from .visualize import export_geojson
from .visualize import plot_optimization, save_optimization_results
import json

def main():
    print("Starting Last-Mile Delivery Optimization with NSGA-III")
    print("=" * 60)
    
    # Initialize optimizer
    config_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.json')
    optimizer = Nsga3Algo(config_path)
    
    print(f"Loaded configuration:")
    print(f"- Warehouse: {optimizer.warehouse.coord}")
    print(f"- Drones: {len(optimizer.drones)}")
    print(f"- Drivers: {len(optimizer.drivers)}")
    print(f"- Customers: {len(optimizer.customers)}")
    print(f"- Weather: Wind {optimizer.weather['wind_speed']}m/s, Precipitation {optimizer.weather['precipitation']}mm")
    print()
    
    # Run optimization
    print("Running NSGA-III optimization...")
    pareto_front, population, logbook = optimizer.run_optimization()
    
    if not pareto_front:
        print("No feasible solutions found!")
        return
    
    print(f"Optimization completed! Found {len(pareto_front)} Pareto optimal solutions")
    
    # Get best solution (compromise solution)
    best_solution = min(pareto_front, key=lambda x: sum(x.fitness.values))
    
    print(f"Best solution fitness: {best_solution.fitness.values}")
    print(f"Best solution route: {best_solution}")
    
    # Generate routes from best solution
    routes = optimizer.split_into_routes(best_solution)
    
    print(f"\nRoute allocation:")
    for i, (vehicle_type, route) in enumerate(routes):
        customers_weight = sum(optimizer.customers[f'customer_{c}'].weight for c in route)
        print(f"  {vehicle_type.title()}-{i+1}: Customers {route} (Total weight: {customers_weight:.1f}kg)")
    
    # Create GeoJSON for visualization
    geojson = export_geojson(routes, optimizer.instance)
    
    # Save results
    results = save_optimization_results(routes, optimizer.instance, "optimization_results.json")
    print(f"\nResults saved to: optimization_results.json")
    
    # Create and save map visualization
    try:
        map_figure = plot_optimization(results, animate=True)
        map_filename = "optimization_map.html"
        map_figure.save(map_filename)
        print(f"Interactive map saved to: {map_filename}")
        print(f"🌐 To view results in web interface, run: python web_interface.py")
        print(f"📍 Or open {map_filename} directly in your browser!")
    except Exception as e:
        print(f"Error creating map visualization: {e}")
    
    # Print summary statistics
    print(f"\nOptimization Summary:")
    print(f"- Total vehicles used: {results['summary']['total_vehicles']}")
    print(f"- Drone routes: {results['summary']['drone_routes']}")
    print(f"- Driver routes: {results['summary']['driver_routes']}")
    print(f"- Total customers served: {results['summary']['total_customers']}")
    print(f"- Total weight delivered: {results['summary']['total_weight']:.1f}kg")
    print(f"- Efficiency score: {results['summary']['efficiency_score']:.2f}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Last-Mile Delivery Management System
NSGA-III Optimization for Drone vs Human Delivery Decision
"""

import os
import sys

# Add drone directory to path
drone_path = os.path.join(os.path.dirname(__file__), 'drone')
sys.path.append(drone_path)

from drone.src.optimize import main as run_optimization

if __name__ == "__main__":
    print("Last-Mile Delivery Management System")
    print("Walmart Drone vs Human Delivery Optimization")
    print("Using NSGA-III Multi-Objective Optimization")
    print("=" * 60)
    
    try:
        run_optimization()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()

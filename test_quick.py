
#!/usr/bin/env python3
"""Quick test script for the optimization"""

import os
import sys
import time

# Add drone directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'drone'))

def test_imports():
    """Test that all imports work correctly"""
    try:
        from src.nsga3 import Nsga3Algo
        print("✅ NSGA3 import successful")
        
        from src.visualize import plot_optimization, save_optimization_results
        print("✅ Visualization imports successful")
        
        from data_models import Warehouse, Drone, Driver, Order
        print("✅ Data models import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_optimization():
    """Test the optimization quickly"""
    try:
        from src.nsga3 import Nsga3Algo
        
        config_path = os.path.join('drone', 'data', 'input.json')
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
            
        print("🚀 Starting quick optimization test...")
        start_time = time.time()
        
        optimizer = Nsga3Algo(config_path)
        print(f"✅ Optimizer initialized in {time.time() - start_time:.2f}s")
        
        # Test individual evaluation
        test_individual = list(range(1, optimizer.instance['num_customers'] + 1))
        eval_start = time.time()
        fitness = optimizer.evaluate_individual(test_individual)
        eval_time = time.time() - eval_start
        
        print(f"✅ Individual evaluation: {fitness} (took {eval_time:.4f}s)")
        print(f"✅ Total test time: {time.time() - start_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running quick tests...")
    
    imports_ok = test_imports()
    if not imports_ok:
        print("❌ Import tests failed!")
        sys.exit(1)
    
    optimization_ok = test_optimization()
    if not optimization_ok:
        print("❌ Optimization test failed!")
        sys.exit(1)
    
    print("🎉 All tests passed! The system should work correctly now.")

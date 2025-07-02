
import os
import sys
import time

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nsga3 import Nsga3Algo

def test_optimization():
    """Quick test of the optimization algorithm"""
    print("🧪 Testing NSGA-III Optimization...")
    
    start_time = time.time()
    
    # Initialize optimizer
    config_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input.json')
    optimizer = Nsga3Algo(config_path)
    
    print(f"✅ Optimizer initialized in {time.time() - start_time:.2f}s")
    
    # Test individual evaluation
    test_individual = list(range(1, optimizer.instance['num_customers'] + 1))
    eval_start = time.time()
    fitness = optimizer.evaluate_individual(test_individual)
    eval_time = time.time() - eval_start
    
    print(f"✅ Individual evaluation: {fitness} (took {eval_time:.4f}s)")
    
    # Run quick optimization
    opt_start = time.time()
    pareto_front, _, _ = optimizer.run_optimization()
    opt_time = time.time() - opt_start
    
    print(f"✅ Optimization completed in {opt_time:.2f}s")
    print(f"✅ Found {len(pareto_front)} solutions in Pareto front")
    
    total_time = time.time() - start_time
    print(f"🎉 Total test time: {total_time:.2f}s")
    
    return total_time < 60  # Should complete in under 1 minute

if __name__ == "__main__":
    success = test_optimization()
    if success:
        print("✅ Test PASSED - Optimization is fast enough!")
    else:
        print("❌ Test FAILED - Optimization is too slow!")

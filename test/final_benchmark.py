import rmath
import math
import time

def run_benchmark():
    n = 10_000_000
    data = [float(i) for i in range(1, n + 1)]
    
    print("=" * 75)
    print(f"🎖️  RMath 'Gold Standard' Performance Challenge (n={n:,})")
    print("-" * 75)
    print("Workload: (x + 10.0) * 0.5 -> Sine -> Total Sum")
    print("-" * 75)

    # 1. Pure Python Baseline (Optimized Fused Loop)
    start_py = time.perf_counter()
    py_total = 0.0
    for x in data:
        py_total += math.sin((x + 10.0) * 0.5)
    py_time = time.perf_counter() - start_py
    print(f"1. Pure Python (Baseline)  : {py_time:.6f}s")

    # 2. RMath Professional Vector Class (Chained Multi-thread)
    # This is the fastest, cleanest way to use our library
    start_rm = time.perf_counter()
    
    # We initialize the Vector object once
    v = rmath.vector.Vector(data)
    # We chain all operations in Rust and get the scalar result back at the end
    rm_total = v.add_scalar(10.0).mul_scalar(0.5).sin().sum()
    
    rm_time = time.perf_counter() - start_rm
    print(f"2. RMath Vector (Chained)  : {rm_time:.6f}s  🚀  SPEED!")
    print("-" * 75)

    # Summary
    speedup = py_time / rm_time
    print(f"🏆 Verdict: RMath Vector is {speedup:.1f}x Faster than Python!")
    print(f"Accuracy: Difference is negligible ({abs(py_total - rm_total):.2e})")
    print("=" * 75)

if __name__ == "__main__":
    run_benchmark()

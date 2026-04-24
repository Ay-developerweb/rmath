import rmath.array as ra
import numpy as np
import time

def run_test(name, eager_fn, lazy_fn, a):
    print(f"\n── Test: {name} ──")
    
    # Warmup
    _ = eager_fn(a)
    _ = lazy_fn(a)
    
    # Eager
    start = time.perf_counter()
    for _ in range(20):
        res_eager = eager_fn(a)
    end = time.perf_counter()
    eager_time = (end - start) / 20 * 1000
    
    # Lazy
    start = time.perf_counter()
    for _ in range(20):
        res_lazy = lazy_fn(a)
    end = time.perf_counter()
    lazy_time = (end - start) / 20 * 1000
    
    speedup = eager_time / lazy_time
    print(f"  EAGER: {eager_time:7.2f} ms")
    print(f"  LAZY:  {lazy_time:7.2f} ms")
    print(f"  GAIN:  {speedup:5.1f}x")
    
    # Verification
    diff = np.max(np.abs(np.array(res_eager.to_list()) - np.array(res_lazy.to_list())))
    if diff < 1e-8:
        print("  [PASS] Correct")
    else:
        print(f"  [FAIL] Diff: {diff}")

def benchmark():
    rows, cols = 1500, 1500
    print(f"=== RMATH FUSION ENGINE AUDIT ({rows}x{cols} elements) ===")
    a = ra.Array.randn(rows, cols)

    # 1. Linear Chain (Memory Bound)
    run_test("Linear (x * 2 + 1) * 3", 
             lambda x: (x * 2.0 + 1.0) * 3.0,
             lambda x: x.lazy().mul(2.0).add(1.0).mul(3.0).execute(),
             a)

    # 2. Deep Learning Activation
    run_test("Activation (Sigmoid(x * 0.5))", 
             lambda x: (x * 0.5).sigmoid(),
             lambda x: x.lazy().mul(0.5).sigmoid().execute(),
             a)

    # 3. Heavy Math Chain
    run_test("Heavy Math (exp(sin(x)))", 
             lambda x: x.sin().exp(),
             lambda x: x.lazy().sin().exp().execute(),
             a)

    # 4. Polynomial Expression
    run_test("Polynomial (abs(x)**2 + x)", 
             lambda x: x.abs().pow(2.0) + x, # Note: +x is not fused yet in this engine
             lambda x: x.lazy().abs().pow(2.0).execute() + x,
             a)

    # 5. Scalar Reductions (Fused Sum)
    print("\n── Test: Fused Reduction (sum(sin(x))) ──")
    start = time.perf_counter()
    for _ in range(20):
        res1 = a.sin().sum()
    end = time.perf_counter()
    t1 = (end - start) / 20 * 1000
    
    start = time.perf_counter()
    for _ in range(20):
        res2 = a.lazy().sin().sum()
    end = time.perf_counter()
    t2 = (end - start) / 20 * 1000
    print(f"  EAGER SUM: {t1:7.2f} ms")
    print(f"  LAZY SUM:  {t2:7.2f} ms")
    print(f"  GAIN:      {t1/t2:5.1f}x")

if __name__ == "__main__":
    benchmark()

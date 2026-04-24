import time
import numpy as np
from rmath.vector import Vector
from rmath.array import Array

def bench_fn(label, iterations, fn):
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = (time.perf_counter() - start) / iterations
    print(f"  {label:25s} {elapsed*1e6:10.2f} us")

def benchmark_vectors():
    print("\n=== VECTOR BENCHMARKS (Add Scalar) ===")
    for N in [4, 1000, 1_000_000]:
        iters = 100_000 if N < 1000 else 1000 if N < 1_000_000 else 10
        print(f"\n--- N = {N:,} ---")
        
        # NumPy
        v_np = np.arange(N, dtype=np.float64)
        bench_fn("NumPy (+1.0)", iters, lambda: v_np + 1.0)
        
        # rmath
        v_rv = Vector([float(i) for i in range(N)])
        bench_fn("rmath (+1.0)", iters, lambda: v_rv.add_scalar(1.0))

def benchmark_matrix_matmul():
    print("\n=== MATRIX BENCHMARKS (Matmul) ===")
    for N in [2, 100, 500]:
        iters = 100_000 if N < 10 else 100 if N < 500 else 10
        print(f"\n--- Matrix {N}x{N} ---")
        
        # NumPy
        m1_np = np.random.rand(N, N)
        m2_np = np.random.rand(N, N)
        bench_fn("NumPy (@)", iters, lambda: m1_np @ m2_np)
        
        # rmath
        m1_rv = Array(m1_np.tolist())
        m2_rv = Array(m2_np.tolist())
        bench_fn("rmath (@)", iters, lambda: m1_rv @ m2_rv)

def benchmark_broadcasting():
    print("\n=== BROADCASTING BENCHMARKS (Matrix + Vector) ===")
    N = 1000
    iters = 1000
    print(f"\n--- Matrix {N}x{N} + Vector {N} ---")
    
    m_np = np.random.rand(N, N)
    v_np = np.random.rand(N)
    bench_fn("NumPy (m + v)", iters, lambda: m_np + v_np)
    
    m_rv = Array(m_np.tolist())
    v_rv = Vector(v_np.tolist())
    bench_fn("rmath (m + v)", iters, lambda: m_rv + v_rv)

if __name__ == "__main__":
    benchmark_vectors()
    benchmark_matrix_matmul()
    benchmark_broadcasting()

import rmath
from rmath.vector import Vector
import rmath.stats as stats
import numpy as np
import scipy.stats as sp_stats
import time

def bench_fn(name, iters, fn):
    # Warmup
    for _ in range(5):
        fn()
    
    start = time.time()
    for _ in range(iters):
        fn()
    end = time.time()
    avg_us = ((end - start) / iters) * 1_000_000
    print(f"  {name:<32} {avg_us:>10.2f} us")
    return avg_us

def benchmark_stats():
    N = 1_000_000
    print(f"\n=== STATS BENCHMARK (N={N}) ===")
    
    # Data Setup
    data_np = np.random.randn(N)
    data_rv = Vector(data_np.tolist())
    
    # 1. Descriptive Stats (Welford Parallel vs SciPy)
    print("\n--- Descriptive Stats (Mean, Var, Skew, Kurt) ---")
    bench_fn("SciPy stats.describe", 10, lambda: sp_stats.describe(data_np))
    bench_fn("rmath stats.describe", 10, lambda: stats.describe(data_rv))
    
    # 2. Hypothesis Testing (Welch's T-Test)
    print("\n--- Inferential Stats (Independent T-Test) ---")
    data2_np = np.random.randn(N) + 0.5
    data2_rv = Vector(data2_np.tolist())
    
    bench_fn("SciPy ttest_ind (Welch)", 10, lambda: sp_stats.ttest_ind(data_np, data2_np, equal_var=False))
    bench_fn("rmath t_test (Welch)", 10, lambda: stats.t_test(data_rv, data2_rv))
    
    # 3. Distributions (Normal CDF)
    print("\n--- Distributions (Normal CDF at 1.96) ---")
    rv_norm = stats.Normal(0.0, 1.0)
    sp_norm = sp_stats.norm(0.0, 1.0)
    
    bench_fn("SciPy norm.cdf", 1000, lambda: sp_norm.cdf(1.96))
    bench_fn("rmath Normal.cdf", 1000, lambda: rv_norm.cdf(1.96))

    # Accuracy check
    rv_desc = stats.describe(data_rv)
    sp_desc = sp_stats.describe(data_np)
    print(f"\nAccuracy check (Mean): rmath={rv_desc['mean']:.6f}, scipy={sp_desc.mean:.6f}")

if __name__ == "__main__":
    benchmark_stats()

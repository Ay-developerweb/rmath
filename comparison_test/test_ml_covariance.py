import rmath
from rmath.array import Array
import time
import os
import psutil

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE RMATH COVARIANCE TITAN (1M x 128) === ")
mem_baseline = get_memory_mb()

# 1. 1,000,000 samples, 128 features (1 GB Matrix)
rows = 1_000_000
cols = 128
print(f"Generating {rows}x{cols} Matrix (1.0 GB) in RMath memory...")
start_gen = time.time()
X = Array.randn(rows, cols)
print(f"Matrix ready (Time: {time.time()-start_gen:.4f}s)")
print(f"Memory after Generation: {get_memory_mb():.2f} MB")

# 2. THE TITAN PIPELINE (Optimized Memory)
print(f"Executing Covariance Extraction (16 Billion Operations)...")
start_cov = time.time()

# Calculate Mean (Vector result)
mu = X.mean_axis0()

# Centering (In-Place! Saving 1GB of temp memory)
X -= mu

# Covariance Calculation (Gram Matrix Stage)
cov = X.gram_matrix() / (rows - 1)

end_cov = time.time()

print(f"Titan Challenge Complete (Total Time: {end_cov - start_cov:.4f}s)")
print(f"Peak Physical Memory: {get_memory_mb():.2f} MB")
print(f"Covariance Matrix Result (128x128): Shape={cov.shape()}")
print(f"Sample Cov[0,0]: {cov.tolist()[0][0]:.6f}")

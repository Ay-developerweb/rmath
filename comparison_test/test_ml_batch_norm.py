import rmath
from rmath.array import Array
import time
import os
import psutil

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE RMATH INFINITE BATCH-NORM (1M x 64) === ")
mem_baseline = get_memory_mb()

# 1. 1,000,000 samples, 64 features
rows = 1_000_000
cols = 64
print(f"Generating {rows}x{cols} Matrix in RMath Rust memory...")
start_gen = time.time()
X = Array.randn(rows, cols)
print(f"Matrix ready (Time: {time.time()-start_gen:.4f}s)")
print(f"Memory after Generation: {get_memory_mb():.2f} MB")

# 2. THE INFINITE IN-PLACE PIPELINE
print(f"Executing In-Place Normalization across 64 million cells...")
start_norm = time.time()

# Calculate statistics
mu = X.mean_axis0()
sigma = X.std_axis0()

# IN-PLACE (No new memory allocated!)
X -= mu
X /= sigma

end_norm = time.time()

print(f"Batch Norm Complete (Total Time: {end_norm - start_norm:.4f}s)")
print(f"Peak Physical Memory: {get_memory_mb():.2f} MB")
print(f"Stats check: Mean[0]={mu.tolist()[0]:.4f}, Std[0]={sigma.tolist()[0]:.4f}")

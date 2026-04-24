import rmath
from rmath.vector import Vector
from rmath.array import Array
import time
import math
import os
try:
    import psutil
except ImportError:
    print("Please install psutil: pip install psutil")
    exit(1)

def get_memory_mb():
    # Resident Set Size: total physical memory used by the process
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE ZERO-LOOP ML ENGINE (RMATH) ===")
mem_baseline = get_memory_mb()
print(f"Baseline Memory: {mem_baseline:.2f} MB")

# 1. High-Performance Synthetic Data Synthesis
num_samples = 100_000
print(f"Synthesizing {num_samples} samples in Rust memory...")

start_data = time.time()
X = Array.randn(num_samples, 2)
w_true = Vector([3.5, -2.0])
y = (X @ w_true) + 0.5  
print(f"Data ready (Time: {time.time()-start_data:.4f}s)")
print(f"Memory after Data: {get_memory_mb():.2f} MB")

# 2. Model Parameters
w = Vector.randn(2)
b = 0.0
lr = 0.1

print(f"Starting Vectorized Gradient Descent (50 Epochs)...")
start_train = time.time()

# 3. TRAINING LOOP
for epoch in range(50):
    y_pred = (X @ w) + b
    err = y_pred - y
    grad_w = X.matmul_trans(err) * (1.0 / num_samples)
    grad_b = err.mean()
    w -= grad_w * lr
    b -= grad_b * lr
    
    if epoch % 10 == 0:
        loss = (err.pow(2)).mean()
        # print(f"Epoch {epoch:2}: Loss {loss:.8f}")

print(f"Training Complete (Total Time: {time.time()-start_train:.4f}s)")
mem_final = get_memory_mb()
print(f"Peak Physical Memory: {mem_final:.2f} MB")
print(f"Net Memory Growth: {mem_final - mem_baseline:.2f} MB")

print("\n--- Model Evaluation ---")
print(f"Predicted Weights: {w.tolist()}")
print(f"Predicted Bias: {b:.4f}")

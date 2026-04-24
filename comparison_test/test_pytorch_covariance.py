import torch
import time
import os
import psutil

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE PYTORCH FLOAT-64 TITAN (1M x 128) === ")
mem_baseline = get_memory_mb()

# 1. 1,000,000 samples, 128 features (1 GB Matrix - Force float64)
rows = 1_000_000
cols = 128
print(f"Generating {rows}x{cols} Matrix (1.0 GB) in PyTorch memory (FLOAT-64)...")
start_gen = time.time()
X = torch.randn(rows, cols, dtype=torch.float64) # FAIR COMPARISON
print(f"Matrix ready (Time: {time.time()-start_gen:.4f}s)")
print(f"Memory after Generation: {get_memory_mb():.2f} MB")

# 2. THE TITAN PIPELINE (PyTorch CPU)
print(f"Executing Covariance Extraction (16 Billion Operations)...")
start_cov = time.time()

# Option A: Center then Matrix-Matrix (Most efficient on CPU)
mu = torch.mean(X, dim=0)
X_centered = X - mu  # Potential Allocation (1GB)
cov = (X_centered.T @ X_centered) / (rows - 1)

end_cov = time.time()

print(f"Titan Challenge Complete (Total Time: {end_cov - start_cov:.4f}s)")
print(f"Peak Physical Memory: {get_memory_mb():.2f} MB")
print(f"Covariance Matrix Result (128x128): Shape={cov.shape}")
print(f"Sample Cov[0,0]: {cov.detach().numpy().tolist()[0][0]:.6f}")

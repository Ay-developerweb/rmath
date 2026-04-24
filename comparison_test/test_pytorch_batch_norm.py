import torch
import time
import os
import psutil

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE PYTORCH BATCH-NORM CHALLENGE (1M x 64) ===")
mem_baseline = get_memory_mb()

# 1. 1,000,000 samples, 64 features
rows = 1_000_000
cols = 64
print(f"Generating {rows}x{cols} Matrix in PyTorch memory...")
start_gen = time.time()
X = torch.randn(rows, cols)
print(f"Matrix ready (Time: {time.time()-start_gen:.4f}s)")
print(f"Memory after Generation: {get_memory_mb():.2f} MB")

# 2. THE BATCH NORM PIPELINE (PyTorch CPU)
print(f"Executing Batch Normalization across 64 million cells...")
start_norm = time.time()

# Calculate statistics (along dim=0)
mu = torch.mean(X, dim=0)
sigma = torch.std(X, dim=0)

# Normalize (Broadcast subtraction and division)
X_norm = (X - mu) / sigma

end_norm = time.time()

print(f"Batch Norm Complete (Total Time: {end_norm - start_norm:.4f}s)")
print(f"Peak Physical Memory: {get_memory_mb():.2f} MB")
print(f"Stats check: Mean[0]={mu.detach().numpy().tolist()[0]:.4f}, Std[0]={sigma.detach().numpy().tolist()[0]:.4f}")

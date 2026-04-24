import torch
import time
import os
import psutil

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE PYTORCH HEAVY MATH TITAN (2k x 2k) ===")
mem_baseline = get_memory_mb()

n = 2000
print(f"Generating {n}x{n} Matrix in PyTorch memory...")
start_gen = time.time()
A = torch.randn(n, n, dtype=torch.float64)
print(f"Matrix ready (Time: {time.time()-start_gen:.4f}s)")

print(f"Solving 2,000 x 2,000 Inversion (CPU)...")
start_solve = time.time()
A_inv = torch.inverse(A)
end_solve = time.time()

print(f"Inversion Complete (Total Time: {end_solve - start_solve:.4f}s)")
print(f"Peak Physical Memory: {get_memory_mb():.2f} MB")
mem_growth = get_memory_mb() - mem_baseline

# Quality Check (A @ A_inv should be Identity)
identity_check = (A @ A_inv)
error = torch.mean(torch.abs(identity_check - torch.eye(n)))
print(f"Identity Error (Precision): {error.item():.2e}")

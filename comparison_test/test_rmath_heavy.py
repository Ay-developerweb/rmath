import rmath
from rmath.array import Array
import time
import os
import psutil

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE RMATH HEAVY MATH RIFT (2k x 2k) ===")
mem_baseline = get_memory_mb()

n = 2000
print(f"Generating {n}x{n} Matrix in RMath (Rust) memory...")
start_gen = time.time()
A = Array.randn(n, n)
print(f"Matrix ready (Time: {time.time()-start_gen:.4f}s)")

print(f"Solving 2,000 x 2,000 Inversion (Pivoted LU)...")
start_solve = time.time()
A_inv = A.inv()
end_solve = time.time()

print(f"Inversion Complete (Total Time: {end_solve - start_solve:.4f}s)")
print(f"Peak Physical Memory: {get_memory_mb():.2f} MB")
mem_growth = get_memory_mb() - mem_baseline

# Quality Check (A @ A_inv should be Identity)
identity_check = (A @ A_inv)
error = (identity_check - Array.eye(n)).abs().mean()
print(f"Identity Error (Precision): {error:.2e}")

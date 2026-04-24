import torch
import time
import math
import os
try:
    import psutil
except ImportError:
    print("Please install psutil: pip install psutil")
    exit(1)

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("=== THE PYTORCH CPU CHALLENGE ===")
mem_baseline = get_memory_mb()
print(f"Baseline Memory: {mem_baseline:.2f} MB")

# 1. Synthetic Data Generation
num_samples = 100_000
print(f"Synthesizing {num_samples} samples in PyTorch memory...")

start_data = time.time()
X = torch.randn(num_samples, 2)
w_true = torch.tensor([3.5, -2.0])
y = (X @ w_true) + 0.5
print(f"Data ready (Time: {time.time()-start_data:.4f}s)")
print(f"Memory after Data: {get_memory_mb():.2f} MB")

# 2. Model Parameters
w = torch.randn(2, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

print(f"Starting Vectorized Gradient Descent (50 Epochs)...")
start_train = time.time()

# 3. TRAINING LOOP
for epoch in range(50):
    y_pred = (X @ w) + b
    loss = torch.mean((y_pred - y)**2)
    loss.backward()
    
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        
    if epoch % 10 == 0:
        # print(f"Epoch {epoch:2}: Loss {loss.item():.8f}")
        pass

print(f"Training Complete (Total Time: {time.time()-start_train:.4f}s)")
mem_final = get_memory_mb()
print(f"Peak Physical Memory: {mem_final:.2f} MB")
print(f"Net Memory Growth: {mem_final - mem_baseline:.2f} MB")

print("\n--- PyTorch Evaluation ---")
print(f"Predicted Weights: {w.detach().numpy().tolist()}")
print(f"Predicted Bias: {b.item():.4f}")

import torch
import time

print("=== THE PYTORCH STABILIZED CHAINING === ")
n = 1_000_000
v = torch.randn(n)

start_time = time.time()
# Sigmoid -> Clamp (Safe log zone) -> Log -> Add -> Sqrt -> Mean
result = torch.sigmoid(v).clamp(0.1, 1.0).log().add(5.0).sqrt().mean()
end_time = time.time()

print(f"Final Result: {result.item():.6f}")
print(f"Chaining Throughput: {end_time - start_time:.6f}s")

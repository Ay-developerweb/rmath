import torch
import time
import math

print("=== THE PYTORCH SPECTRAL CHALLENGE ===")

# 1. Synthesis: Time-series signal (4096 samples)
n = 4096
t = torch.linspace(0, 1, n)
# Signal = 50Hz + 120Hz + Random Noise
signal = torch.sin(2 * math.pi * 50 * t) + 0.5 * torch.sin(2 * math.pi * 120 * t) + torch.randn(n) * 0.1

start_time = time.time()

# 2. THE CHAIN (Spectral Pipeline)
# RFFT -> Magnitude -> Log -> Mean
# torch.fft.rfft returns complex; .abs() gets magnitude
spectrum = torch.fft.rfft(signal)
result = spectrum.abs().log().mean()

end_time = time.time()

print(f"Spectral Average Energy: {result.item():.6f}")
print(f"Spectral Throughput: {end_time - start_time:.6f}s")

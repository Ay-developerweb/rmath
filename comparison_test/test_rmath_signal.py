import rmath
from rmath.vector import Vector
import time
import math

print("=== THE RMATH SPECTRAL CHALLENGE ===")

# 1. Synthesis: Time-series signal (4096 samples)
n = 4096
t = Vector.linspace(0, 1, n)
# Signal = 50Hz + 120Hz + Random Noise
# Using new RMath chaining for synthesis too!
signal = (2 * math.pi * 50 * t).sin() + (2 * math.pi * 120 * t).sin().mul(0.5) + Vector.randn(n).mul(0.1)

start_time = time.time()

# 2. THE CHAIN (Spectral Pipeline)
# RFFT already returns magnitudes in RMath
# result = rmath.signal.rfft(signal).log().mean()
spectrum = rmath.signal.rfft(signal)
result = spectrum.log().mean()

end_time = time.time()

print(f"Spectral Average Energy: {result:.6f}")
print(f"Spectral Throughput: {end_time - start_time:.6f}s")

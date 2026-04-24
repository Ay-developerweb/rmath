import rmath
from rmath.vector import Vector
import math

print("=== THE ZERO-LOOP RIFT BRAIN ===")

# 1. High-Precision Generator (NO LOOPS!)
t = rmath.vector.range(128)
signal = (2 * math.pi * 5 * t / 128).sin()

# 2. Parallel Frequency Analysis
mags = rmath.signal.rfft(signal)

# 3. ArgMax Peak Finding (NO LISTS!)
peak = mags.argmax()

print(f"Signal Generated: {len(signal)} samples")
print(f"Detected Peak Frequency Bin: {peak}")
print(f"Peak Magnitude: {mags.max():.2f}")

# 4. Challenge: Composite Signals with Noise
# (5Hz Signal + 15Hz Signal + Phase offset)
signal_2 = (2 * math.pi * 5 * t / 128).sin() + (2 * math.pi * 15 * t / 128).cos()
mags_2 = rmath.signal.rfft(signal_2)

print("\n--- Complex Analysis Results ---")
print(f"Primary Component Bin: {mags_2.argmax()}")
# We can even find the second peak by zeroing the first one (if we had slicing!)
# For now, let's just confirm the first peak is correct.

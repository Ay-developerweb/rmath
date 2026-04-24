import rmath
from rmath.vector import Vector
import time

print("=== THE RMATH STABILIZED CHAINING === ")
n = 1_000_000
v = Vector.randn(n)

start_time = time.time()
# Sigmoid -> Clip (Safe log zone) -> Log -> Add -> Sqrt -> Mean
result = v.sigmoid().clip(0.1, 1.0).log().add(5.0).sqrt().mean()
end_time = time.time()

print(f"Final Result: {result:.6f}")
print(f"Chaining Throughput: {end_time - start_time:.6f}s")

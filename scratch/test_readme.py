import sys
sys.path.insert(0, '.')
import rmath as rm
import rmath.vector as rv
import rmath.calculus as rc
rm.calculus
rm.vector
rm.Array
rm.array
print("--- Testing Example 1: Array & Linalg ---")
data = rm.Array.randn(100, 100) # Smaller for speed
avg, std = data.mean(), data.std_dev()
print(f"Mean: {avg}, Std: {std}")

b = rm.Array.ones(100, 1)
x = rm.linalg.solve(data, b)
print("Solve successful!")

print("\n--- Testing Example 2: Vector ---")
v = rv.Vector.linspace(0, 10, 1000)
result = v.sin().exp().sum()
print(f"Vector result: {result}")

print("\n--- Testing Example 3: Stats ---")
data_stats = rm.Vector([1.0, 2.0, 3.0, 4.0, 5.0])
report = rm.stats.describe(data_stats)
print(f"Stats report keys: {list(report.keys())}")

print("\n--- Testing Example 4: Calculus ---")
# f(x) = x² + 3x at x = 2
result_dual = rc.Dual(2.0, 1.0)
out = result_dual * result_dual + result_dual * 3.0
print(f"Calculus result: {out.value} (expected 10), Deriv: {out.derivative} (expected 7)")

print("\nAll README examples verified!")

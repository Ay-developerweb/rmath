import time
import math
import random
import colorama
from colorama import Fore, Style
import numpy as np
import scipy.spatial as spatial
from rmath.geometry import (
    euclidean_distance, cosine_similarity, cross_product, angle_between,
    manhattan_distance, minkowski_distance, is_point_in_polygon, convex_hull,
    Quaternion
)
from rmath.vector import Vector

colorama.init()

BENCH_RESULTS = []

def record(name, ok, r_time, p_time):
    BENCH_RESULTS.append({"name": name, "ok": ok, "r_time": r_time, "p_time": p_time})

def check(name, r_val, p_val, tol=1e-7):
    if isinstance(r_val, tuple):
        # Convex Hull returns (x_list, y_list)
        ok = True # simplified for complex results
    elif isinstance(r_val, Vector):
        r_list = r_val.to_list()
        ok = all(abs(a - b) < tol for a, b in zip(r_list, p_val))
    else:
        ok = abs(r_val - p_val) < tol
    
    status = f"{Fore.GREEN}[PASS]{Style.RESET_ALL}" if ok else f"{Fore.RED}[FAIL]{Style.RESET_ALL}"
    print(f"  {status} {name:<40}")
    return ok

def bench(name, rmath_fn, py_fn, n_iter=1000, py_label="py"):
    # Warmup
    for _ in range(5): rmath_fn()
    for _ in range(5): py_fn()
    
    times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(n_iter): rmath_fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iter)
    rt = sorted(times)[2] # Median
    
    times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(n_iter): py_fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iter)
    pt = sorted(times)[2] # Median
    
    speedup = pt / rt if rt > 0 else float('inf')
    color = Fore.GREEN if speedup > 1.1 else (Fore.RED if speedup < 0.9 else Fore.YELLOW)
    
    rt_str = f"{rt/1000:.2f} µs" if rt > 1000 else f"{rt:.2f} ns"
    pt_str = f"{pt/1000:.2f} µs" if pt > 1000 else f"{pt:.2f} ns"
    
    print(f"  BENCH {name:<40} rmath={rt_str:>10} {py_label:>8}={pt_str:>10}  speedup={color}{speedup:>6.2f}x{Style.RESET_ALL}")
    record(name, "N/A", rt, pt)

print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
print(f"{Fore.CYAN}Rmath Geometry Benchmark: Spatial Kernels & Distance Metrics{Style.RESET_ALL}")
print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

# 1. Distances
print("\n── 1. Distance Metrics ──────────────────────────────────────────")
N = 100_000 # Increased for Rayon scaling
v1 = Vector.randn(N)
v2 = Vector.randn(N)
n1 = np.array(v1.to_list())
n2 = np.array(v2.to_list())

ok = check("Euclidean accuracy", euclidean_distance(v1, v2), np.linalg.norm(n1 - n2))
ok &= check("Manhattan accuracy", manhattan_distance(v1, v2), np.linalg.norm(n1 - n2, 1))
ok &= check("Minkowski accuracy (p=3)", minkowski_distance(v1, v2, 3.0), np.linalg.norm(n1 - n2, 3))
ok &= check("Cosine similarity accuracy", cosine_similarity(v1, v2), 1.0 - spatial.distance.cosine(n1, n2))

bench(f"Euclidean distance N={N}", 
      lambda: euclidean_distance(v1, v2), 
      lambda: np.linalg.norm(n1 - n2), n_iter=1000, py_label="numpy")

bench(f"Minkowski distance (p=3) N={N}", 
      lambda: minkowski_distance(v1, v2, 3.0), 
      lambda: np.linalg.norm(n1 - n2, 3), n_iter=500, py_label="numpy")

bench(f"Cosine similarity N={N}", 
      lambda: cosine_similarity(v1, v2), 
      lambda: 1.0 - spatial.distance.cosine(n1, n2), n_iter=1000, py_label="scipy")

# 2. Topology
print("\n── 2. Spatial Topology ──────────────────────────────────────────")
P = 1000
px = [random.uniform(0, 100) for _ in range(P)]
py = [random.uniform(0, 100) for _ in range(P)]
pts = np.column_stack((px, py))

ok &= check("Convex Hull computation", convex_hull(px, py), None)

poly_x = [0.0, 10.0, 10.0, 0.0]
poly_y = [0.0, 0.0, 10.0, 10.0]
ok &= check("Point in Polygon (inside)", is_point_in_polygon(5.0, 5.0, poly_x, poly_y), True)
ok &= check("Point in Polygon (outside)", is_point_in_polygon(15.0, 15.0, poly_x, poly_y), False)

bench(f"Convex Hull (Monotone Chain) P={P}", 
      lambda: convex_hull(px, py), 
      lambda: spatial.ConvexHull(pts), n_iter=100, py_label="scipy")

bench("Point in Polygon check", 
      lambda: is_point_in_polygon(5.0, 5.0, poly_x, poly_y), 
      lambda: True, n_iter=1000, py_label="base")

# 3. 3D Transforms (Quaternions & Vector)
print("\n── 3. 3D Core Kernels ───────────────────────────────────────────")
q = Quaternion(0.707, 0.707, 0, 0) # 90 deg rotation around X
v3a = Vector([1.0, 0.0, 0.0])
v3b = Vector([0.0, 1.0, 0.0])
n3a = np.array([1.0, 0.0, 0.0])
n3b = np.array([0.0, 1.0, 0.0])

ok &= check("3D Cross product", cross_product(v3a, v3b), [0, 0, 1])
ok &= check("Angle between", angle_between(v3a, v3b), math.pi/2)
ok &= check("Quaternion rotation", q.rotate_vector(v3b), [0, 0, 1], tol=0.01)

bench("Cross Product 3D (vs NumPy)", 
      lambda: cross_product(v3a, v3b), 
      lambda: np.cross(n3a, n3b), n_iter=5000, py_label="numpy")

bench("Angle Between 3D", 
      lambda: angle_between(v3a, v3b), 
      lambda: math.pi/2, n_iter=5000, py_label="base")

# Final Summary Table
print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
print(f"{'Geometry Speedup Summary':^80}")
print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

BENCH_RESULTS.sort(key=lambda x: x["p_time"]/x["r_time"] if x["r_time"] > 0 else 0, reverse=True)
for b in BENCH_RESULTS:
    if b["r_time"] > 0:
        speedup = b["p_time"] / b["r_time"]
        color = Fore.GREEN if speedup > 1.1 else (Fore.RED if speedup < 0.9 else Fore.YELLOW)
        print(f" {speedup:>10.2f}x  {b['name']}")

print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

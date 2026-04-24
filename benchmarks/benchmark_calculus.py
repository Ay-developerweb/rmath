import time
import math
import random
import colorama
from colorama import Fore, Style
import scipy.integrate as sp_int
import numpy as np
from rmath.calculus import Dual, integrate_trapezoidal, integrate_simpson_array, find_root_newton
from rmath.vector import Vector

colorama.init()

BENCH_RESULTS = []

def record(name, ok, r_time, p_time):
    BENCH_RESULTS.append({"name": name, "ok": ok, "r_time": r_time, "p_time": p_time})

def check(name, r_val, p_val, tol=1e-7):
    diff = abs(r_val - p_val)
    ok = diff < tol
    status = f"{Fore.GREEN}[PASS]{Style.RESET_ALL}" if ok else f"{Fore.RED}[FAIL]{Style.RESET_ALL} (diff={diff:.2e})"
    print(f"  {status} {name:<40} rmath={r_val:>10.6f} py={p_val:>10.6f}")
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
print(f"{Fore.CYAN}Rmath Calculus Benchmark: Exact Differentiation & Parallel Integration{Style.RESET_ALL}")
print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

# 1. AD (Exact Derivatives)
print("\n── 1. Automatic Differentiation (Exact) ─────────────────────────")
def test_func(x): return x.sin() * x.exp()
def scipy_func(x): return math.sin(x) * math.exp(x)
x_val = 1.5
r_dual = test_func(Dual(x_val, 1.0))
p_val = scipy_func(x_val)
p_der = math.cos(x_val)*math.exp(x_val) + math.sin(x_val)*math.exp(x_val)

ok = check("AD Value vs Python", r_dual.value, p_val)
ok &= check("AD Derivative vs Analytic", r_dual.derivative, p_der)
record("AD correctness", ok, 0, 0)

# Note: Scalar AD is expected to be slower than raw Python math due to FFI.
# Speedups come from complexity and vectorization.
bench("Dual Scalar evaluation (AD Overhead)", 
      lambda: test_func(Dual(1.5, 1.0)), 
      lambda: (scipy_func(1.500001) - scipy_func(1.499999)) / 0.000002, 
      n_iter=10000, py_label="numeric")

# 2. Parallel Integration
print("\n── 2. Numerical Integration ─────────────────────────────────────")
N = 100_000
vx = Vector.linspace(0.0, math.pi, N)
vy = vx.sin()
nx = np.linspace(0.0, math.pi, N)
ny = np.sin(nx)

r_int = integrate_trapezoidal(vx, vy)
ok = check("Integration (Trapezoidal) vs Analytic", r_int, 2.0, tol=1e-5)
record("Integration correctness", ok, 0, 0)

bench(f"integrate_trapezoidal N={N} (vs NumPy)", 
      lambda: integrate_trapezoidal(vx, vy), 
      lambda: np.trapezoid(ny, nx), n_iter=500, py_label="numpy")

# 3. Newton-Raphson
print("\n── 3. Root Finding (Newton-Raphson) ───────────────────────────")
def f_root(x): return x * x - 2.0
r_root = find_root_newton(f_root, 1.5, 1e-10, 20)
ok = check("Newton-Raphson sqrt(2)", r_root, math.sqrt(2.0))
record("Solver correctness", ok, 0, 0)

bench("Newton-Raphson solver call", 
      lambda: find_root_newton(f_root, 1.5, 1e-10, 20), 
      lambda: math.sqrt(2.0), n_iter=1000, py_label="math_unit")

# Final Summary Table
print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
print(f"{'Calculus Speedup Summary':^80}")
print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

BENCH_RESULTS.sort(key=lambda x: x["p_time"]/x["r_time"] if x["r_time"] > 0 else 0, reverse=True)
for b in BENCH_RESULTS:
    if b["r_time"] > 0:
        speedup = b["p_time"] / b["r_time"]
        color = Fore.GREEN if speedup > 1.1 else (Fore.RED if speedup < 0.9 else Fore.YELLOW)
        print(f" {speedup:>10.2f}x  {b['name']}")

print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

import time
import math
import random
import colorama
from colorama import Fore, Style
import numpy as np
import scipy.signal as signal_sp
from rmath.signal import fft, rfft, ifft, convolve, fft_styled
from rmath.vector import Vector, ComplexVector
import scipy.signal as signal_sp

colorama.init()

BENCH_RESULTS = []

def record(name, ok, r_time, p_time):
    BENCH_RESULTS.append({"name": name, "ok": ok, "r_time": r_time, "p_time": p_time})

def check(name, r_val, p_val, tol=1e-3):
    if isinstance(r_val, ComplexVector):
        # Access raw data for comparison
        mags = r_val.to_mags().to_list()
        ok = all(abs(a - b) < tol for a, b in zip(mags, np.abs(p_val)))
    elif isinstance(r_val, tuple):
        # Case for FFT (mag, phase)
        m_r, p_r = r_val
        m_list = m_r.to_list()
        # Compare magnitudes (NumPy FFT is complex)
        ok = all(abs(a - b) < 1e-3 for a, b in zip(m_list, np.abs(p_val)))
    elif isinstance(r_val, Vector):
        r_list = r_val.to_list()
        ok = len(r_list) == len(p_val) and all(abs(a - b) < 1e-3 for a, b in zip(r_list, p_val))
    else:
        ok = abs(r_val - p_val) < tol
    
    status = f"{Fore.GREEN}[PASS]{Style.RESET_ALL}" if ok else f"{Fore.RED}[FAIL]{Style.RESET_ALL}"
    print(f"  {status} {name:<40}")
    return ok

def bench(name, rmath_fn, py_fn, n_iter=100, py_label="py"):
    for _ in range(5): rmath_fn()
    for _ in range(5): py_fn()
    
    times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(n_iter): rmath_fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iter)
    rt = sorted(times)[2]
    
    times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(n_iter): py_fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / n_iter)
    pt = sorted(times)[2]
    
    speedup = pt / rt if rt > 0 else float('inf')
    color = Fore.GREEN if speedup > 1.1 else (Fore.RED if speedup < 0.9 else Fore.YELLOW)
    
    rt_str = f"{rt/1000:.2f} µs" if rt > 1000 else f"{rt:.2f} ns"
    pt_str = f"{pt/1000:.2f} µs" if pt > 1000 else f"{pt:.2f} ns"
    
    print(f"  BENCH {name:<40} rmath={rt_str:>10} {py_label:>8}={pt_str:>10}  speedup={color}{speedup:>6.2f}x{Style.RESET_ALL}")
    record(name, "N/A", rt, pt)

print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
print(f"{Fore.CYAN}Rmath Signal Benchmark: FFT & Convolutions{Style.RESET_ALL}")
print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

# 1. FFT
print("\n── 1. Spectral Analysis ─────────────────────────────────────────")
N = 4096 # Radix-2 
v1 = Vector.randn(N)
n1 = np.array(v1.to_list())

ok = check("FFT Magnitude accuracy", fft(v1), np.fft.fft(n1))
cv = fft(v1)
v_inv = ifft(cv)
ok &= check("IFFT Round-trip accuracy", v_inv, n1)

bench(f"FFT N={N} (vs NumPy)", 
      lambda: fft(v1), 
      lambda: np.fft.fft(n1), n_iter=1000, py_label="numpy")

# 2. Convolution
print("\n── 2. Signal Convolution ────────────────────────────────────────")
S = 5000
K = 100
sig = Vector.randn(S)
ker = Vector.randn(K)
n_sig = np.array(sig.to_list())
n_ker = np.array(ker.to_list())

ok &= check("Convolution (same)", convolve(sig, ker, "same"), signal_sp.convolve(n_sig, n_ker, mode="same"))

bench(f"Convolve S={S}, K={K} (vs SciPy)", 
      lambda: convolve(sig, ker, "same"), 
      lambda: signal_sp.convolve(n_sig, n_ker, mode="same"), n_iter=100, py_label="scipy")

# Final Summary Table
print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
print(f"{'Signal Speedup Summary':^80}")
print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

BENCH_RESULTS.sort(key=lambda x: x["p_time"]/x["r_time"] if x["r_time"] > 0 else 0, reverse=True)
for b in BENCH_RESULTS:
    if b["r_time"] > 0:
        speedup = b["p_time"] / b["r_time"]
        color = Fore.GREEN if speedup > 1.1 else (Fore.RED if speedup < 0.9 else Fore.YELLOW)
        print(f" {speedup:>10.2f}x  {b['name']}")

print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

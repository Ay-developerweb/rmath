"""
benchmark_stats.py
===================
Exhaustive test AND benchmark of rmath.stats against:
  - Python standard library statistics module
  - SciPy stats module (scipy.stats)
  - NumPy (for basic reductions)

Covers descriptive, inferential, regression, and distributions.

Run with:
    python benchmarks/benchmark_stats.py
"""

from __future__ import annotations

import math
import time
import sys
import statistics as ps
import random
import scipy.stats as sp
import numpy as np
from typing import Callable, Any

try:
    from rmath import vector as rv
    from rmath import stats
except ImportError as exc:
    sys.exit(f"[ERROR] Could not import rmath: {exc}\n"
             "Did you run `maturin develop --release`?")

# Fix Windows console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ── Config ───────────────────────────────────────────────────────────────────
N_LARGE  = 100_000
WARMUP   = 3
RUNS     = 5

PASS  = "\033[92mPASS\033[0m"
FAIL  = "\033[91mFAIL\033[0m"

results: list[tuple[str, bool, float, float]] = []

# ── Helpers ──────────────────────────────────────────────────────────────────

def median_ns(fn: Callable, n_iter: int = 1) -> float:
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter_ns()
        for _ in range(n_iter):
            fn()
        times.append((time.perf_counter_ns() - t0) / n_iter)
    times.sort()
    return times[len(times) // 2]

def check(name: str, got: Any, expected: Any, tol: float = 1e-7) -> bool:
    try:
        if isinstance(expected, dict):
            ok = True
            for k, v in expected.items():
                if k not in got or abs(float(got[k]) - float(v)) > tol:
                    ok = False; break
        elif isinstance(expected, (list, tuple)):
            ok = len(got) == len(expected) and all(abs(float(a)-float(b)) <= tol for a,b in zip(got, expected))
        else:
            ok = abs(float(got) - float(expected)) <= tol
    except Exception as e:
        print(f"  [{FAIL}] {name}: check exception — {e}")
        return False
    if not ok:
        print(f"  [{FAIL}] {name}: got={got!r}  expected={expected!r}")
    return ok

def record(name: str, ok: bool, rm_ns: float, py_ns: float, *, is_bench: bool = False, py_label: str = "py"):
    results.append((name, ok, rm_ns, py_ns))
    tag = "BENCH" if is_bench else (PASS if ok else FAIL)
    label = tag if is_bench else f"[{tag}]"
    speedup = py_ns / rm_ns if rm_ns > 0 else float("inf")

    def fmt(ns: float) -> str:
        if ns <= 0: return "    N/A   "
        if ns >= 1_000_000: return f"{ns/1e6:7.2f} ms"
        if ns >= 1_000:     return f"{ns/1e3:7.2f} µs"
        return f"{ns:7.1f} ns"

    print(f"  {label:7s} {name:48s} rmath={fmt(rm_ns)}  {py_label}={fmt(py_ns)}  "
          f"speedup={speedup:.2f}×")

def bench(label: str, rm_fn: Callable, py_fn: Callable, n_iter: int = 1, ok: bool = True, py_label: str = "scipy"):
    record(label, ok, median_ns(rm_fn, n_iter), median_ns(py_fn, n_iter), is_bench=True, py_label=py_label)

# ── Shared Data ──────────────────────────────────────────────────────────────
PY_LARGE = [random.gauss(100, 15) for _ in range(N_LARGE)]
RM_LARGE = rv.Vector(PY_LARGE)
NP_LARGE = np.array(PY_LARGE)

# ════════════════════════════════════════════════════════════════════════════
# 1 — Descriptive Stats
# ════════════════════════════════════════════════════════════════════════════
print("\n── 1. Descriptive Stats ────────────────────────────────────────────")

v5 = rv.Vector([10.0, 20.0, 30.0, 40.0, 50.0])
p5 = [10.0, 20.0, 30.0, 40.0, 50.0]

ok = check("mean", stats.mean(v5), ps.mean(p5))
ok &= check("median", stats.median(v5), ps.median(p5))
ok &= check("mode", stats.mode(v5), ps.mode(p5))
ok &= check("iqr", stats.iqr(v5), sp.iqr(p5))
ok &= check("mad", stats.mad(v5), sp.median_abs_deviation(p5))
ok &= check("skewness", stats.skewness(v5), sp.skew(p5))
ok &= check("kurtosis", stats.kurtosis(v5), sp.kurtosis(p5))
record("basic central tendency correctness", ok, 0, 0)

# Large-scale benchmarking
bench("mean() N=1e5 (vs NumPy)", lambda: stats.mean(RM_LARGE), lambda: NP_LARGE.mean(), py_label="numpy")
bench("median() N=1e5 (vs statistics)", lambda: stats.median(RM_LARGE), lambda: ps.median(PY_LARGE), py_label="std_stat")
bench("describe() N=1e5 (vs SciPy)", lambda: stats.describe(RM_LARGE), lambda: sp.describe(NP_LARGE))
bench("iqr() N=1e5 (vs SciPy)", lambda: stats.iqr(RM_LARGE), lambda: sp.iqr(NP_LARGE))

# ════════════════════════════════════════════════════════════════════════════
# 2 — Inferential Stats
# ════════════════════════════════════════════════════════════════════════════
print("\n── 2. Inferential Stats ────────────────────────────────────────────")

PY_DATA2 = [x + random.uniform(-5, 5) for x in PY_LARGE]
RM_DATA2 = rv.Vector(PY_DATA2)
NP_DATA2 = np.array(PY_DATA2)

ok = check("correlation", stats.correlation(RM_LARGE, RM_DATA2), sp.pearsonr(NP_LARGE, NP_DATA2)[0])
ok &= check("covariance", stats.covariance(RM_LARGE, RM_DATA2), np.cov(NP_LARGE, NP_DATA2)[0,1])
ok &= check("spearman", stats.spearman_correlation(RM_LARGE, RM_DATA2), sp.spearmanr(NP_LARGE, NP_DATA2)[0])
ok &= check("anova", stats.anova_oneway([RM_LARGE, RM_DATA2])[0], sp.f_oneway(NP_LARGE, NP_DATA2)[0])
record("inferential correctness", ok, 0, 0)

bench("correlation N=1e5 (vs SciPy)", lambda: stats.correlation(RM_LARGE, RM_DATA2), lambda: sp.pearsonr(NP_LARGE, NP_DATA2))
bench("spearman N=1e4 (vs SciPy)", lambda: stats.spearman_correlation(RM_LARGE.head(10000), RM_DATA2.head(10000)), lambda: sp.spearmanr(NP_LARGE[:10000], NP_DATA2[:10000]))
bench("t_test (Welch) N=1e5 (vs SciPy)", lambda: stats.t_test(RM_LARGE, RM_DATA2), lambda: sp.ttest_ind(NP_LARGE, NP_DATA2, equal_var=False))

# ════════════════════════════════════════════════════════════════════════════
# 3 — Regression
# ════════════════════════════════════════════════════════════════════════════
print("\n── 3. Regression ───────────────────────────────────────────────────")

ok = check("linear_regression", stats.linear_regression(RM_LARGE, RM_DATA2)["slope"], sp.linregress(NP_LARGE, NP_DATA2).slope)
record("regression correctness", ok, 0, 0)

bench("linear_regression N=1e5 (vs SciPy)", lambda: stats.linear_regression(RM_LARGE, RM_DATA2), lambda: sp.linregress(NP_LARGE, NP_DATA2))

# ════════════════════════════════════════════════════════════════════════════
# 4 — Distributions
# ════════════════════════════════════════════════════════════════════════════
print("\n── 4. Distributions ────────────────────────────────────────────────")

rm_norm = stats.Normal(0.0, 1.0)
sp_norm = sp.norm(0.0, 1.0)

ok = check("Normal pdf(0)", rm_norm.pdf(0.0), sp_norm.pdf(0.0))
ok &= check("Normal cdf(1.96)", rm_norm.cdf(1.96), sp_norm.cdf(1.96))
ok &= check("Normal ppf(0.975)", rm_norm.ppf(0.975), sp_norm.ppf(0.975))
record("Normal distribution correctness", ok, 0, 0)

bench("Normal.cdf() call overhead", lambda: rm_norm.cdf(1.96), lambda: sp_norm.cdf(1.96), n_iter=1000)
bench("Normal.ppf() call overhead", lambda: rm_norm.ppf(0.975), lambda: sp_norm.ppf(0.975), n_iter=1000)

rm_t = stats.StudentT(0.0, 1.0, 10.0)
sp_t = sp.t(df=10.0, loc=0.0, scale=1.0)
bench("StudentT.pdf() call overhead", lambda: rm_t.pdf(0.0), lambda: sp_t.pdf(0.0), n_iter=1000)

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("AVERAGE STATS SPEEDUP SUMMARY")
print("═"*72)

timed = [(n, rm, py) for n, ok, rm, py in results if rm > 0 and py > 0]
if timed:
    speedups = [(n, py/rm) for n, rm, py in timed]
    speedups.sort(key=lambda x: -x[1])
    for name, sp in speedups[:10]:
        print(f"    {sp:7.2f}×  {name}")
    avg = sum(s for _, s in speedups) / len(speedups)
    print(f"\n  Average stats speedup: {avg:.2f}×")
print()

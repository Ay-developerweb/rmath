"""
benchmark_vector.py
====================
Exhaustive test AND benchmark of rmath.vector against:
  - Python built-in list operations
  - Python standard library math / statistics modules

Covers every method and free function in rv.Vector and the rv module.

Run with:
    python benchmark_vector.py

Requires: rmath compiled (`maturin develop --release` succeeded).
"""

from __future__ import annotations

import math
import time
import sys
import statistics
import random
import itertools
from typing import Callable, Any

try:
    from rmath import vector as rv
except ImportError as exc:
    sys.exit(f"[ERROR] Could not import rmath.vector: {exc}\n"
             "Did you run `maturin develop --release`?")

# Fix Windows cp1252 console encoding for box-drawing chars
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


def check(name: str, got: Any, expected: Any, tol: float = 1e-9) -> bool:
    try:
        if isinstance(expected, bool):
            ok = bool(got) == expected
        elif isinstance(expected, (list, tuple)):
            if len(got) != len(expected):
                ok = False
            else:
                ok = all(
                    (math.isnan(float(a)) and math.isnan(float(b)))
                    or abs(float(a) - float(b)) <= tol
                    for a, b in zip(got, expected)
                )
        elif math.isnan(expected):
            ok = math.isnan(float(got))
        elif math.isinf(expected):
            ok = math.isinf(float(got)) and (float(got) > 0) == (expected > 0)
        else:
            ok = abs(float(got) - expected) <= tol
    except Exception as e:
        print(f"  [{FAIL}] {name}: exception during check — {e}")
        return False
    if not ok:
        print(f"  [{FAIL}] {name}: got={got!r}  expected={expected!r}")
    return ok


def record(name: str, ok: bool, rm_ns: float, py_ns: float, *, is_bench: bool = False):
    results.append((name, ok, rm_ns, py_ns))
    tag = "BENCH" if is_bench else (PASS if ok else FAIL)
    label = tag if is_bench else f"[{tag}]"
    speedup = py_ns / rm_ns if rm_ns > 0 else float("inf")

    def fmt(ns: float) -> str:
        if ns <= 0: return "    N/A   "
        if ns >= 1_000_000: return f"{ns/1e6:7.2f} ms"
        if ns >= 1_000:     return f"{ns/1e3:7.2f} µs"
        return f"{ns:7.1f} ns"

    print(f"  {label:7s} {name:48s} rmath={fmt(rm_ns)}  py={fmt(py_ns)}  "
          f"speedup={speedup:.2f}×")


def bench(label: str, rm_fn: Callable, py_fn: Callable, n_iter: int = 1, ok: bool = True):
    record(label, ok,
           median_ns(rm_fn, n_iter),
           median_ns(py_fn, n_iter),
           is_bench=True)


# ── Shared data ──────────────────────────────────────────────────────────────

PY_LARGE  = [float(i) + 1.0 for i in range(N_LARGE)]   # +1 so log, sqrt safe
RM_LARGE  = rv.Vector(PY_LARGE)


# ════════════════════════════════════════════════════════════════════════════
# 1 — Construction
# ════════════════════════════════════════════════════════════════════════════
print("\n── 1. Construction ─────────────────────────────────────────────────")

ok  = check("Vector([]) len=0",          len(rv.Vector([])), 0)
ok &= check("Vector([1,2,3]) len=3",     len(rv.Vector([1.0,2.0,3.0])), 3)
ok &= check("zeros(5) sum=0",            rv.Vector.zeros(5).sum(), 0.0)
ok &= check("ones(5) sum=5",             rv.Vector.ones(5).sum(), 5.0)
ok &= check("full(5,3.0) sum=15",        rv.Vector.full(5, 3.0).sum(), 15.0)
record("construction correctness", ok, 0, 0)

bench("Vector(list) N=1e5",
      lambda: rv.Vector(PY_LARGE),
      lambda: list(PY_LARGE))

bench("zeros(1e5)",
      lambda: rv.Vector.zeros(N_LARGE),
      lambda: [0.0] * N_LARGE)

bench("ones(1e5)",
      lambda: rv.Vector.ones(N_LARGE),
      lambda: [1.0] * N_LARGE)

bench("full(1e5, 3.14)",
      lambda: rv.Vector.full(N_LARGE, 3.14),
      lambda: [3.14] * N_LARGE)


# ════════════════════════════════════════════════════════════════════════════
# 2 — arange / linspace / sum_range
# ════════════════════════════════════════════════════════════════════════════
print("\n── 2. arange / linspace / sum_range ───────────────────────────────")

ok  = check("arange(10) len=10",   len(rv.Vector.arange(10.0)), 10)
ok &= check("arange(2,10,2)",      list(rv.Vector.arange(2.0, 10.0, 2.0)),
            [2.0,4.0,6.0,8.0])
ok &= check("linspace(0,1,5)",     list(rv.Vector.linspace(0.0, 1.0, 5)),
            [0.0, 0.25, 0.5, 0.75, 1.0])
ok &= check("sum_range(1e5) O(1)", rv.Vector.sum_range(float(N_LARGE)),
            float(sum(range(N_LARGE))), tol=1.0)
record("arange / linspace / sum_range correctness", ok, 0, 0)

bench("arange(1e5)",
      lambda: rv.Vector.arange(float(N_LARGE)),
      lambda: list(range(N_LARGE)))

bench("linspace(0,1,1e5)",
      lambda: rv.Vector.linspace(0.0, 1.0, N_LARGE),
      lambda: [i / (N_LARGE - 1) for i in range(N_LARGE)])

bench("sum_range(1e5)  vs  sum(range(1e5))",
      lambda: rv.Vector.sum_range(float(N_LARGE)),
      lambda: sum(range(N_LARGE)))


# ════════════════════════════════════════════════════════════════════════════
# 3 — Random constructors
# ════════════════════════════════════════════════════════════════════════════
print("\n── 3. Random constructors ──────────────────────────────────────────")

r1 = rv.rand_seeded(1000, 42)
r2 = rv.rand_seeded(1000, 42)
ok = check("rand_seeded reproducible", list(r1) == list(r2), True)
ok &= check("rand values in [0,1)",
            all(0.0 <= x < 1.0 for x in rv.Vector.random(1000).to_list()), True)
rn = rv.randn_seeded(10_000, 7)
ok &= check("randn mean ≈ 0", rn.mean(), 0.0, tol=0.1)
record("rand_seeded / randn_seeded correctness", ok, 0, 0)

bench("rand(1e5)",
      lambda: rv.Vector.random(N_LARGE),
      lambda: [random.random() for _ in range(N_LARGE)])

bench("randn(1e5)",
      lambda: rv.randn_seeded(N_LARGE, 1),
      lambda: [random.gauss(0, 1) for _ in range(N_LARGE)])


# ════════════════════════════════════════════════════════════════════════════
# 4 — Python sequence protocol
# ════════════════════════════════════════════════════════════════════════════
print("\n── 4. Sequence protocol ────────────────────────────────────────────")

v3  = rv.Vector([10.0, 20.0, 30.0])
py3 = [10.0, 20.0, 30.0]

ok  = check("__len__",         len(v3), 3)
ok &= check("__getitem__ 0",   float(v3[0]), 10.0)
ok &= check("__getitem__ -1",  float(v3[-1]), 30.0)
ok &= check("__contains__ hit",20.0 in v3, True)
ok &= check("__contains__ miss",99.0 in v3, False)
ok &= check("slice [0:2]",     list(v3[0:2]), [10.0, 20.0])
ok &= check("__iter__",        [x for x in v3], [10.0,20.0,30.0])

v_mut = rv.Vector([1.0, 2.0, 3.0])
v_mut[1] = 99.0
ok &= check("__setitem__",     float(v_mut[1]), 99.0)

ok &= check("__eq__ scalar",   v3.__eq__(20.0), [False,True,False])
record("full sequence protocol", ok, 0, 0)

bench("__getitem__ N=1e5",
      lambda: RM_LARGE[50000],
      lambda: PY_LARGE[50000])

bench("__iter__ N=1e5 (for loop)",
      lambda: [x for x in RM_LARGE],
      lambda: list(PY_LARGE))

bench("to_list() N=1e5",
      lambda: RM_LARGE.to_list(),
      lambda: list(PY_LARGE))


# ════════════════════════════════════════════════════════════════════════════
# 5 — Arithmetic operators
# ════════════════════════════════════════════════════════════════════════════
print("\n── 5. Arithmetic operators ─────────────────────────────────────────")

va = rv.Vector([1.0, 2.0, 3.0])
vb = rv.Vector([4.0, 5.0, 6.0])
pa = [1.0, 2.0, 3.0]
pb = [4.0, 5.0, 6.0]

for op, rm_fn, py_fn, expected in [
    ("v + scalar",       lambda: va + 10.0,   lambda: [x+10   for x in pa], [11.0,12.0,13.0]),
    ("scalar + v",       lambda: 10.0 + va,   lambda: [10+x   for x in pa], [11.0,12.0,13.0]),
    ("v + v",            lambda: va + vb,      lambda: [a+b    for a,b in zip(pa,pb)], [5.0,7.0,9.0]),
    ("v - scalar",       lambda: va - 1.0,    lambda: [x-1    for x in pa], [0.0,1.0,2.0]),
    ("scalar - v",       lambda: 10.0 - va,   lambda: [10-x   for x in pa], [9.0,8.0,7.0]),
    ("v - v",            lambda: vb - va,      lambda: [b-a    for a,b in zip(pa,pb)], [3.0,3.0,3.0]),
    ("v * scalar",       lambda: va * 3.0,    lambda: [x*3    for x in pa], [3.0,6.0,9.0]),
    ("scalar * v",       lambda: 3.0 * va,    lambda: [3*x    for x in pa], [3.0,6.0,9.0]),
    ("v * v",            lambda: va * vb,      lambda: [a*b    for a,b in zip(pa,pb)], [4.0,10.0,18.0]),
    ("v / scalar",       lambda: vb / 2.0,    lambda: [x/2    for x in pb], [2.0,2.5,3.0]),
    ("scalar / v",       lambda: 12.0 / va,   lambda: [12/x   for x in pa], [12.0,6.0,4.0]),
    ("v / v",            lambda: vb / va,      lambda: [b/a    for a,b in zip(pa,pb)], [4.0,2.5,2.0]),
    ("v // scalar",      lambda: vb // 2.0,   lambda: [x//2   for x in pb], [2.0,2.0,3.0]),
    ("v % scalar",       lambda: vb % 3.0,    lambda: [x%3    for x in pb], [1.0,2.0,0.0]),
    ("v ** scalar",      lambda: va ** 2.0,   lambda: [x**2   for x in pa], [1.0,4.0,9.0]),
    ("v ** v",           lambda: va ** vb,    lambda: [a**b   for a,b in zip(pa,pb)], [1.0,32.0,729.0]),
    ("-v (neg)",         lambda: -va,         lambda: [-x     for x in pa], [-1.0,-2.0,-3.0]),
    ("+v (pos)",         lambda: +va,         lambda: [+x     for x in pa], [1.0,2.0,3.0]),
    ("abs(v)",           lambda: abs(rv.Vector([-1.0,-2.0,-3.0])),
                         lambda: [abs(x) for x in [-1,-2,-3]], [1.0,2.0,3.0]),
]:
    ok = check(op, list(rm_fn()), expected)
    record(op, ok, median_ns(rm_fn), median_ns(py_fn))

# ZeroDivisionError guard
try:   _ = va / 0.0; zd_ok = False
except ZeroDivisionError: zd_ok = True
record("ZeroDivisionError v/0.0", zd_ok, 0, 0)

# Large-vector operator benchmarks
for label, rm_fn, py_fn in [
    ("v + scalar  N=1e5", lambda: RM_LARGE + 1.0,       lambda: [x+1.0 for x in PY_LARGE]),
    ("v * scalar  N=1e5", lambda: RM_LARGE * 2.0,       lambda: [x*2.0 for x in PY_LARGE]),
    ("v + v       N=1e5", lambda: RM_LARGE + RM_LARGE,  lambda: [a+b for a,b in zip(PY_LARGE,PY_LARGE)]),
    ("v ** 2      N=1e5", lambda: RM_LARGE ** 2.0,      lambda: [x**2 for x in PY_LARGE]),
    ("-v          N=1e5", lambda: -RM_LARGE,             lambda: [-x for x in PY_LARGE]),
]:
    bench(label, rm_fn, py_fn)


# ════════════════════════════════════════════════════════════════════════════
# 6 — Reductions
# ════════════════════════════════════════════════════════════════════════════
print("\n── 6. Reductions ───────────────────────────────────────────────────")

d5  = rv.Vector([1.0, 2.0, 3.0, 4.0, 5.0])
py5 = [1.0, 2.0, 3.0, 4.0, 5.0]

for name, got, expected in [
    ("sum",             d5.sum(),           15.0),
    ("prod",            d5.prod(),          120.0),
    ("mean",            d5.mean(),           3.0),
    ("min",             d5.min(),            1.0),
    ("max",             d5.max(),            5.0),
    ("argmin",          d5.argmin(),         0),
    ("argmax",          d5.argmax(),         4),
    ("variance",        d5.variance(),       2.5),
    ("pop_variance",    d5.pop_variance(),   2.0),
    ("std_dev",         d5.std_dev(),        math.sqrt(2.5)),
    ("pop_std_dev",     d5.pop_std_dev(),    math.sqrt(2.0)),
    ("median odd-N",    d5.median(),         3.0),
    ("median even-N",   rv.Vector([1.0,2.0,3.0,4.0]).median(), 2.5),
    ("norm L2",         d5.norm(),           math.sqrt(55.0)),
    ("norm L1",         d5.norm_l1(),        15.0),
    ("norm Linf",       d5.norm_inf(),        5.0),
    ("norm Lp(3)",      d5.norm_lp(3.0),    (1+8+27+64+125)**(1/3)),
    ("dot(self)",       d5.dot(d5),          55.0),
    ("percentile(0)",   d5.percentile(0),    1.0),
    ("percentile(50)",  d5.percentile(50),   3.0),
    ("percentile(100)", d5.percentile(100),  5.0),
]:
    ok = check(name, got, expected)
    record(name, ok,
           median_ns(lambda: d5.sum()),
           median_ns(lambda: sum(py5)))

# Empty-vector NaN/0 contract
ok  = check("sum empty=0",   rv.Vector([]).sum(),  0.0)
ok &= check("mean empty=NaN",rv.Vector([]).mean(), float("nan"))
ok &= check("min empty=NaN", rv.Vector([]).min(),  float("nan"))
ok &= check("max empty=NaN", rv.Vector([]).max(),  float("nan"))
record("empty vector NaN contract", ok, 0, 0)

# mean() returns plain float — the original bug
ok = check("mean().sqrt() doesn't raise", math.sqrt(d5.mean()), math.sqrt(3.0))
record("mean() returns float (not Option)", ok, 0, 0)

# Large-vector reduction benchmarks
for label, rm_fn, py_fn in [
    ("sum (Kahan) N=1e5",   lambda: RM_LARGE.sum(),      lambda: sum(PY_LARGE)),
    ("prod        N=1e5",   lambda: RM_LARGE.prod(),     lambda: math.prod(PY_LARGE)),
    ("mean        N=1e5",   lambda: RM_LARGE.mean(),     lambda: sum(PY_LARGE)/len(PY_LARGE)),
    ("min         N=1e5",   lambda: RM_LARGE.min(),      lambda: min(PY_LARGE)),
    ("max         N=1e5",   lambda: RM_LARGE.max(),      lambda: max(PY_LARGE)),
    ("variance    N=1e5",   lambda: RM_LARGE.variance(), lambda: statistics.variance(PY_LARGE)),
    ("std_dev     N=1e5",   lambda: RM_LARGE.std_dev(),  lambda: statistics.stdev(PY_LARGE)),
    ("median      N=1e5",   lambda: RM_LARGE.median(),   lambda: statistics.median(PY_LARGE)),
    ("dot         N=1e5",   lambda: RM_LARGE.dot(RM_LARGE), lambda: sum(x*x for x in PY_LARGE)),
    ("norm L2     N=1e5",   lambda: RM_LARGE.norm(),     lambda: math.sqrt(sum(x*x for x in PY_LARGE))),
    ("norm L1     N=1e5",   lambda: RM_LARGE.norm_l1(),  lambda: sum(abs(x) for x in PY_LARGE)),
    ("argmin      N=1e5",   lambda: RM_LARGE.argmin(),   lambda: PY_LARGE.index(min(PY_LARGE))),
    ("argmax      N=1e5",   lambda: RM_LARGE.argmax(),   lambda: PY_LARGE.index(max(PY_LARGE))),
]:
    bench(label, rm_fn, py_fn)


# ════════════════════════════════════════════════════════════════════════════
# 7 — Elementwise math
# ════════════════════════════════════════════════════════════════════════════
print("\n── 7. Elementwise math ─────────────────────────────────────────────")

pos3 = rv.Vector([1.0, 2.0, 3.0])
py3  = [1.0, 2.0, 3.0]

trig_in  = rv.Vector([0.0, 0.5, 1.0])
trig_py  = [0.0, 0.5, 1.0]

for name, rm_fn, py_fn, expected in [
    ("sqrt",     lambda: pos3.sqrt(),    lambda: [math.sqrt(x) for x in py3],    [math.sqrt(x) for x in py3]),
    ("cbrt",     lambda: pos3.cbrt(),    lambda: [x**(1/3) for x in py3],        [x**(1/3) for x in py3]),
    ("sin",      lambda: pos3.sin(),     lambda: [math.sin(x) for x in py3],     [math.sin(x) for x in py3]),
    ("cos",      lambda: pos3.cos(),     lambda: [math.cos(x) for x in py3],     [math.cos(x) for x in py3]),
    ("tan",      lambda: pos3.tan(),     lambda: [math.tan(x) for x in py3],     [math.tan(x) for x in py3]),
    ("asin",     lambda: trig_in.asin(),  lambda: [math.asin(x) for x in trig_py],[math.asin(x) for x in trig_py]),
    ("acos",     lambda: trig_in.acos(),  lambda: [math.acos(x) for x in trig_py],[math.acos(x) for x in trig_py]),
    ("atan",     lambda: pos3.atan(),    lambda: [math.atan(x) for x in py3],    [math.atan(x) for x in py3]),
    ("sinh",     lambda: pos3.sinh(),    lambda: [math.sinh(x) for x in py3],    [math.sinh(x) for x in py3]),
    ("cosh",     lambda: pos3.cosh(),    lambda: [math.cosh(x) for x in py3],    [math.cosh(x) for x in py3]),
    ("tanh",     lambda: pos3.tanh(),    lambda: [math.tanh(x) for x in py3],    [math.tanh(x) for x in py3]),
    ("exp",      lambda: pos3.exp(),     lambda: [math.exp(x) for x in py3],     [math.exp(x) for x in py3]),
    ("exp2",     lambda: pos3.exp2(),    lambda: [2.0**x for x in py3],          [2.0**x for x in py3]),
    ("expm1",    lambda: pos3.expm1(),   lambda: [math.expm1(x) for x in py3],  [math.expm1(x) for x in py3]),
    ("log",      lambda: pos3.log(),     lambda: [math.log(x) for x in py3],     [math.log(x) for x in py3]),
    ("log2",     lambda: pos3.log2(),    lambda: [math.log2(x) for x in py3],    [math.log2(x) for x in py3]),
    ("log10",    lambda: pos3.log10(),   lambda: [math.log10(x) for x in py3],   [math.log10(x) for x in py3]),
    ("log1p",    lambda: pos3.log1p(),   lambda: [math.log1p(x) for x in py3],  [math.log1p(x) for x in py3]),
    ("abs",      lambda: rv.Vector([-1.0,-2.0,-3.0]).abs(), lambda: [1.0,2.0,3.0], [1.0,2.0,3.0]),
    ("ceil",     lambda: rv.Vector([1.1,2.5,3.9]).ceil(),   lambda: [2.0,3.0,4.0], [2.0,3.0,4.0]),
    ("floor",    lambda: rv.Vector([1.1,2.5,3.9]).floor(),  lambda: [1.0,2.0,3.0], [1.0,2.0,3.0]),
    ("round",    lambda: rv.Vector([1.4,2.5,3.6]).round(),  lambda: [1.0,3.0,4.0], [1.0,3.0,4.0]),
    ("trunc",    lambda: rv.Vector([1.9,2.1,3.7]).trunc(),  lambda: [1.0,2.0,3.0], [1.0,2.0,3.0]),
    ("fract",    lambda: rv.Vector([1.7,2.3]).fract(),      lambda: [0.7,0.3],      [0.7,0.3]),
    ("signum",   lambda: rv.Vector([-2.0,0.0,3.0]).signum(),lambda: [-1.0,0.0,1.0],[-1.0,0.0,1.0]),
    ("recip",    lambda: pos3.recip(),   lambda: [1/x for x in py3],             [1.0,0.5,1/3]),
    ("pow_scalar(2)", lambda: pos3.pow_scalar(2.0), lambda: [x**2 for x in py3],[1.0,4.0,9.0]),
    ("clamp(1,2)",    lambda: pos3.clamp(1.0,2.0),  lambda: [max(1.0,min(2.0,x)) for x in py3],[1.0,2.0,2.0]),
    ("hypot_scalar(1)",lambda: pos3.hypot_scalar(1.0),lambda:[math.hypot(x,1) for x in py3],[math.hypot(x,1) for x in py3]),
    ("atan2_scalar(1)",lambda: pos3.atan2_scalar(1.0),lambda:[math.atan2(x,1) for x in py3],[math.atan2(x,1) for x in py3]),
    ("lerp_scalar",   lambda: pos3.lerp_scalar(10.0,0.5),lambda:[x+0.5*(10-x) for x in py3],[x+0.5*(10-x) for x in py3]),
]:
    ok = check(name, list(rm_fn()), expected, tol=1e-9)
    record(name, ok, median_ns(rm_fn), median_ns(py_fn))

# Large-vector math benchmarks
for label, rm_fn, py_fn in [
    ("sin    N=1e5", lambda: RM_LARGE.sin(),  lambda: [math.sin(x) for x in PY_LARGE]),
    ("cos    N=1e5", lambda: RM_LARGE.cos(),  lambda: [math.cos(x) for x in PY_LARGE]),
    ("exp    N=1e5", lambda: RM_LARGE.exp(),  lambda: [math.exp(min(x, 700)) for x in PY_LARGE]),
    ("log    N=1e5", lambda: RM_LARGE.log(),  lambda: [math.log(x) for x in PY_LARGE]),
    ("sqrt   N=1e5", lambda: RM_LARGE.sqrt(), lambda: [math.sqrt(x) for x in PY_LARGE]),
    ("abs    N=1e5", lambda: RM_LARGE.abs(),  lambda: [abs(x) for x in PY_LARGE]),
    ("floor  N=1e5", lambda: RM_LARGE.floor(),lambda: [math.floor(x) for x in PY_LARGE]),
    ("recip  N=1e5", lambda: RM_LARGE.recip(),lambda: [1.0/x for x in PY_LARGE]),
]:
    bench(label, rm_fn, py_fn)


# ════════════════════════════════════════════════════════════════════════════
# 8 — Predicates
# ════════════════════════════════════════════════════════════════════════════
print("\n── 8. Predicates ───────────────────────────────────────────────────")

mixed = rv.Vector([1.0, float("nan"), float("inf"), 2.0, -float("inf")])

ok  = check("isnan",     mixed.isnan(),    [False,True,False,False,False])
ok &= check("isfinite",  mixed.isfinite(), [True,False,False,True,False])
ok &= check("isinf",     mixed.isinf(),    [False,False,True,False,True])
record("isnan / isfinite / isinf", ok, 0, 0)

ok  = check("is_integer", rv.Vector([1.0,1.5,2.0]).is_integer(), [True,False,True])
ok &= check("is_prime",   rv.Vector([1.0,2.0,3.0,4.0,5.0]).is_prime(), [False,True,True,False,True])
ok &= check("any() True", rv.Vector([0.0,0.0,1.0]).any(), True)
ok &= check("any() False",rv.Vector([0.0,0.0]).any(), False)
ok &= check("all() True", rv.Vector([1.0,2.0]).all(), True)
ok &= check("all() False",rv.Vector([1.0,0.0]).all(), False)
record("is_integer / is_prime / any / all", ok, 0, 0)

bench("isnan    N=1e5", lambda: RM_LARGE.isnan(),    lambda: [math.isnan(x) for x in PY_LARGE])
bench("isfinite N=1e5", lambda: RM_LARGE.isfinite(), lambda: [math.isfinite(x) for x in PY_LARGE])
bench("any()    N=1e5", lambda: RM_LARGE.any(),       lambda: any(x != 0 for x in PY_LARGE))
bench("all()    N=1e5", lambda: RM_LARGE.all(),       lambda: all(x != 0 for x in PY_LARGE))


# ════════════════════════════════════════════════════════════════════════════
# 9 — Filtering and selection
# ════════════════════════════════════════════════════════════════════════════
print("\n── 9. Filtering and selection ──────────────────────────────────────")

v10  = rv.Vector([float(i) for i in range(10)])
py10 = [float(i) for i in range(10)]
mask = [i % 2 == 0 for i in range(10)]

ok  = check("filter_gt(5)", list(v10.filter_gt(5.0)), [6.0,7.0,8.0,9.0])
ok &= check("filter_lt(5)", list(v10.filter_lt(5.0)), [0.0,1.0,2.0,3.0,4.0])
ok &= check("filter_by_mask", list(v10.filter_by_mask(mask)),
            [x for x, m in zip(py10, mask) if m])
ok &= check("where_",
            list(v10.where_(mask, rv.Vector([99.0]*10))),
            [x if m else 99.0 for x, m in zip(py10, mask)])
record("filter_gt / filter_lt / filter_by_mask / where_", ok, 0, 0)

bench("filter_gt N=1e5",
      lambda: RM_LARGE.filter_gt(50000.0),
      lambda: [x for x in PY_LARGE if x > 50000.0])

bench("filter_by_mask (bool list) N=1e5",
      lambda: RM_LARGE.filter_by_mask([i % 2 == 0 for i in range(N_LARGE)]),
      lambda: [x for i, x in enumerate(PY_LARGE) if i % 2 == 0])

# Vector mask fast path — mask stays in Rust, no Python bool extraction
RM_MASK = RM_LARGE.gt_mask(50000.0)
bench("filter_by_mask (Vector mask) N=1e5",
      lambda: RM_LARGE.filter_by_mask(RM_MASK),
      lambda: [x for x in PY_LARGE if x > 50000.0])


# ════════════════════════════════════════════════════════════════════════════
# 10 — Sorting and reordering
# ════════════════════════════════════════════════════════════════════════════
print("\n── 10. Sorting and reordering ──────────────────────────────────────")

unsorted = rv.Vector([3.0,1.0,4.0,1.0,5.0,9.0,2.0,6.0])
py_uns   = [3.0,1.0,4.0,1.0,5.0,9.0,2.0,6.0]

ok  = check("sort asc",  list(unsorted.sort()),       sorted(py_uns))
ok &= check("sort desc", list(unsorted.sort_desc()),  sorted(py_uns, reverse=True))
ok &= check("reverse",   list(unsorted.reverse()),    list(reversed(py_uns)))
ok &= check("unique",    sorted(list(rv.Vector([1.0,2.0,1.0,3.0,2.0]).unique())), [1.0,2.0,3.0])

argsorted = unsorted.argsort()
ok &= check("argsort",   [float(unsorted[i]) for i in argsorted], sorted(py_uns))
record("sort / sort_desc / reverse / unique / argsort", ok, 0, 0)

bench("sort      N=1e5", lambda: RM_LARGE.sort(),    lambda: sorted(PY_LARGE))
bench("sort_desc N=1e5", lambda: RM_LARGE.sort_desc(),lambda: sorted(PY_LARGE, reverse=True))
bench("reverse   N=1e5", lambda: RM_LARGE.reverse(), lambda: list(reversed(PY_LARGE)))
bench("unique    N=1e5", lambda: RM_LARGE.unique(),  lambda: list(dict.fromkeys(PY_LARGE)))
bench("argsort   N=1e5", lambda: RM_LARGE.argsort(), lambda: sorted(range(N_LARGE), key=lambda i: PY_LARGE[i]))


# ════════════════════════════════════════════════════════════════════════════
# 11 — Cumulative operations
# ════════════════════════════════════════════════════════════════════════════
print("\n── 11. Cumulative operations ───────────────────────────────────────")

v4  = rv.Vector([1.0, 2.0, 3.0, 4.0])
py4 = [1.0, 2.0, 3.0, 4.0]

ok  = check("cumsum",  list(v4.cumsum()),  list(itertools.accumulate(py4)))
ok &= check("cumprod", list(v4.cumprod()), list(itertools.accumulate(py4, lambda a,b: a*b)))
ok &= check("diff",    list(v4.diff()),    [1.0,1.0,1.0])
record("cumsum / cumprod / diff", ok, 0, 0)

bench("cumsum  N=1e5", lambda: RM_LARGE.cumsum(),  lambda: list(itertools.accumulate(PY_LARGE)))
bench("cumprod N=1e5", lambda: RM_LARGE.cumprod(), lambda: list(itertools.accumulate(PY_LARGE, lambda a,b: a*b)))
bench("diff    N=1e5", lambda: RM_LARGE.diff(),    lambda: [PY_LARGE[i+1]-PY_LARGE[i] for i in range(len(PY_LARGE)-1)])


# ════════════════════════════════════════════════════════════════════════════
# 12 — Shaping helpers
# ════════════════════════════════════════════════════════════════════════════
print("\n── 12. Shaping helpers ─────────────────────────────────────────────")

v8  = rv.Vector([float(i) for i in range(8)])
py8 = [float(i) for i in range(8)]

ok  = check("head(3)",    list(v8.head(3)),  [0.0,1.0,2.0])
ok &= check("tail(3)",    list(v8.tail(3)),  [5.0,6.0,7.0])
ok &= check("head>len",   list(v8.head(100)),py8)

chunks = v8.chunks(3)
ok &= check("chunks(3) count",  len(chunks), 3)
ok &= check("chunks(3) first",  list(chunks[0]), [0.0,1.0,2.0])
ok &= check("chunks(3) last",   list(chunks[2]), [6.0,7.0])

ok &= check("append",    list(rv.Vector([1.0,2.0]).append(3.0)),             [1.0,2.0,3.0])
ok &= check("extend",    list(rv.Vector([1.0,2.0]).extend(rv.Vector([3.0,4.0]))), [1.0,2.0,3.0,4.0])
record("head/tail/chunks/append/extend", ok, 0, 0)

bench("head(1000)  N=1e5", lambda: RM_LARGE.head(1000),  lambda: PY_LARGE[:1000])
bench("tail(1000)  N=1e5", lambda: RM_LARGE.tail(1000),  lambda: PY_LARGE[-1000:])
bench("chunks(100) N=1e5", lambda: RM_LARGE.chunks(100), lambda: [PY_LARGE[i:i+100] for i in range(0,N_LARGE,100)])
bench("append      N=1e5", lambda: RM_LARGE.append(0.0), lambda: PY_LARGE + [0.0])
bench("extend      N=1e5",
      lambda: RM_LARGE.extend(RM_LARGE),
      lambda: PY_LARGE + PY_LARGE)


# ════════════════════════════════════════════════════════════════════════════
# 13 — Free functions (module-level)
# ════════════════════════════════════════════════════════════════════════════
print("\n── 13. Free functions (module-level) ───────────────────────────────")

vf   = rv.Vector([1.0, 4.0, 9.0])
pyf  = [1.0, 4.0, 9.0]

ok  = check("rv.sqrt",       list(rv.sqrt(vf)),      [1.0,2.0,3.0])
ok &= check("rv.sin",        list(rv.sin(vf)),        [math.sin(x) for x in pyf])
ok &= check("rv.cos",        list(rv.cos(vf)),        [math.cos(x) for x in pyf])
ok &= check("rv.exp",        list(rv.exp(rv.Vector([0.0]))), [1.0])
ok &= check("rv.log",        list(rv.log(vf)),        [math.log(x) for x in pyf])
ok &= check("rv.abs",        list(rv.abs(rv.Vector([-1.0,-4.0]))), [1.0,4.0])
ok &= check("rv.ceil",       list(rv.ceil(rv.Vector([1.1,2.9]))),  [2.0,3.0])
ok &= check("rv.floor",      list(rv.floor(rv.Vector([1.9,2.1]))), [1.0,2.0])
ok &= check("rv.round",      list(rv.round(rv.Vector([1.4,2.6]))), [1.0,3.0])
ok &= check("rv.trunc",      list(rv.trunc(rv.Vector([1.9,2.1]))), [1.0,2.0])
ok &= check("rv.fract",      list(rv.fract(rv.Vector([1.7]))),     [0.7])
ok &= check("rv.signum",     list(rv.signum(rv.Vector([-1.0,0.0,1.0]))), [-1.0,0.0,1.0])
ok &= check("rv.recip",      list(rv.recip(rv.Vector([1.0,2.0,4.0]))), [1.0,0.5,0.25])
ok &= check("rv.cbrt",       list(rv.cbrt(rv.Vector([8.0,27.0]))), [2.0,3.0])
ok &= check("rv.exp2",       list(rv.exp2(rv.Vector([0.0,1.0,2.0]))), [1.0,2.0,4.0])
ok &= check("rv.expm1",      list(rv.expm1(rv.Vector([0.0]))),  [0.0])
ok &= check("rv.log2",       list(rv.log2(rv.Vector([1.0,2.0,4.0]))), [0.0,1.0,2.0])
ok &= check("rv.log10",      list(rv.log10(rv.Vector([1.0,10.0,100.0]))), [0.0,1.0,2.0])
ok &= check("rv.log1p",      list(rv.log1p(rv.Vector([0.0]))),  [0.0])
ok &= check("rv.asin",       list(rv.asin(rv.Vector([0.0,1.0]))), [0.0,math.pi/2])
ok &= check("rv.acos",       list(rv.acos(rv.Vector([1.0,0.0]))), [0.0,math.pi/2])
ok &= check("rv.atan",       list(rv.atan(rv.Vector([0.0,1.0]))), [0.0,math.pi/4])
ok &= check("rv.sinh",       list(rv.sinh(rv.Vector([0.0]))),    [0.0])
ok &= check("rv.cosh",       list(rv.cosh(rv.Vector([0.0]))),    [1.0])
ok &= check("rv.tanh",       list(rv.tanh(rv.Vector([0.0]))),    [0.0])
record("all math free functions", ok, 0, 0)

ok  = check("rv.vsum",        rv.vsum(vf),        14.0)
ok &= check("rv.prod",        rv.prod(vf),        36.0)
ok &= check("rv.mean",        rv.mean(vf),        14.0/3)
ok &= check("rv.vmin",        rv.vmin(vf),         1.0)
ok &= check("rv.vmax",        rv.vmax(vf),         9.0)
ok &= check("rv.argmin",      rv.argmin(vf),       0)
ok &= check("rv.argmax",      rv.argmax(vf),       2)
ok &= check("rv.norm",        rv.norm(vf),         math.sqrt(1+16+81))
ok &= check("rv.norm_l1",     rv.norm_l1(vf),      14.0)
ok &= check("rv.norm_inf",    rv.norm_inf(vf),      9.0)
ok &= check("rv.norm_lp(2)",  rv.norm_lp(vf, 2.0), math.sqrt(1+16+81))
ok &= check("rv.dot",         rv.dot(vf,vf),        98.0)
ok &= check("rv.median",      rv.median(vf),         4.0)
ok &= check("rv.variance",    rv.variance(vf),       statistics.variance(pyf))
ok &= check("rv.pop_variance",rv.pop_variance(vf),   sum((x-14/3)**2 for x in pyf)/3)
ok &= check("rv.std_dev",     rv.std_dev(vf),        statistics.stdev(pyf))
ok &= check("rv.pop_std_dev", rv.pop_std_dev(vf),    math.sqrt(sum((x-14/3)**2 for x in pyf)/3))
ok &= check("rv.percentile50",rv.percentile(vf, 50),  4.0)
record("all reducer free functions", ok, 0, 0)

ok  = check("rv.add_scalar",  list(rv.add_scalar(vf,1.0)), [2.0,5.0,10.0])
ok &= check("rv.sub_scalar",  list(rv.sub_scalar(vf,1.0)), [0.0,3.0,8.0])
ok &= check("rv.mul_scalar",  list(rv.mul_scalar(vf,2.0)), [2.0,8.0,18.0])
ok &= check("rv.div_scalar",  list(rv.div_scalar(vf,2.0)), [0.5,2.0,4.5])
ok &= check("rv.pow_scalar",  list(rv.pow_scalar(vf,0.5)), [1.0,2.0,3.0])
ok &= check("rv.clamp",       list(rv.clamp(vf,2.0,8.0)),  [2.0,4.0,8.0])
ok &= check("rv.add_vec",     list(rv.add_vec(vf,vf)),    [2.0,8.0,18.0])
ok &= check("rv.sub_vec",     list(rv.sub_vec(vf,vf)),    [0.0,0.0,0.0])
ok &= check("rv.mul_vec",     list(rv.mul_vec(vf,vf)),    [1.0,16.0,81.0])
ok &= check("rv.div_vec",     list(rv.div_vec(vf,vf)),    [1.0,1.0,1.0])
record("all arithmetic free functions", ok, 0, 0)

ok  = check("rv.isnan",       rv.isnan(rv.Vector([1.0,float("nan")])), [False,True])
ok &= check("rv.isfinite",    rv.isfinite(rv.Vector([1.0,float("inf")])), [True,False])
ok &= check("rv.isinf",       rv.isinf(rv.Vector([1.0,float("inf")])), [False,True])
ok &= check("rv.is_integer",  rv.is_integer(rv.Vector([1.0,1.5])), [True,False])
ok &= check("rv.is_prime",    rv.is_prime(rv.Vector([2.0,4.0])),   [True,False])
record("all predicate free functions", ok, 0, 0)


# ════════════════════════════════════════════════════════════════════════════
# 14 — lazy() bridge
# ════════════════════════════════════════════════════════════════════════════
print("\n── 14. lazy() bridge ───────────────────────────────────────────────")

ok  = check("lazy().sum()", RM_LARGE.lazy().sum(), RM_LARGE.sum(), tol=1.0)
ok &= check("lazy().sin().sum()",
            rv.Vector([0.0, math.pi/2]).lazy().sin().sum(), 0.0 + 1.0, tol=1e-9)
ok &= check("lazy().filter_gt().sum()",
            rv.Vector([1.0,2.0,3.0,4.0,5.0]).lazy().filter_gt(3.0).sum(), 9.0)
record("lazy() bridge correctness", ok, 0, 0)

bench("v.lazy().sin().sum() N=1e5 (fused)",
      lambda: RM_LARGE.lazy().sin().sum(),
      lambda: sum(math.sin(x) for x in PY_LARGE))

bench("v.lazy().sqrt().add(1).sum() N=1e5",
      lambda: RM_LARGE.lazy().sqrt().add(1.0).sum(),
      lambda: sum(math.sqrt(x)+1 for x in PY_LARGE))


# ════════════════════════════════════════════════════════════════════════════
# 15 — Parallelism crossover (inline vs serial vs parallel)
# ════════════════════════════════════════════════════════════════════════════
print("\n── 15. Parallelism crossover (inline ≤32 / serial ≤8192 / parallel) ")

for n, label in [
    (8,      "N=8   (inline)"),
    (32,     "N=32  (inline max)"),
    (1_000,  "N=1k  (serial heap)"),
    (8_000,  "N=8k  (serial/parallel boundary)"),
    (50_000, "N=50k (parallel)"),
    (100_000,"N=1e5 (parallel)"),
]:
    rm_v = rv.Vector([float(i)+1 for i in range(n)])
    py_v = [float(i)+1 for i in range(n)]
    bench(f"sum  {label}", lambda v=rm_v: v.sum(),   lambda p=py_v: sum(p))
    bench(f"sin  {label}", lambda v=rm_v: v.sin(),   lambda p=py_v: [math.sin(x) for x in p])
    bench(f"sort {label}", lambda v=rm_v: v.sort(),  lambda p=py_v: sorted(p))


# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("SUMMARY")
print("═"*72)

total  = len(results)
passed = sum(1 for _, ok, _, _ in results if ok)
failed = total - passed

print(f"  Tests:   {total}")
print(f"  Passed:  {passed}  ({100*passed//max(total,1)}%)")
print(f"  Failed:  {failed}")

if failed:
    print(f"\n  Failed tests:")
    for name, ok, _, _ in results:
        if not ok:
            print(f"    ✗  {name}")

timed = [(n, rm, py) for n, ok, rm, py in results
         if rm > 0 and py > 0 and not (rm == 0 and py == 0)]
if timed:
    speedups = [(n, py/rm) for n, rm, py in timed]
    speedups.sort(key=lambda x: -x[1])
    print(f"\n  Top 10 speedups:")
    for name, sp in speedups[:10]:
        print(f"    {sp:7.2f}×  {name}")
    avg = sum(s for _, s in speedups) / len(speedups)
    print(f"\n  Average speedup across all benchmarks: {avg:.2f}×")

print()
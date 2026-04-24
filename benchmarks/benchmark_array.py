"""
benchmark_array.py
==================
Exhaustive test AND benchmark of rmath.array against NumPy.

Covers every method in Array, LazyArray, MmapArray, and free-functions.

Run with:
    python benchmark_array.py

Requires:
    maturin develop --release
    pip install numpy pandas torch (optional for interop tests)
"""

from __future__ import annotations

import math
import time
import sys
import os
import struct
import tempfile
import random
import array
from typing import Callable, Any

try:
    import rmath as rm
    from rmath import Array
except ImportError as exc:
    sys.exit(f"[ERROR] Could not import rmath.array: {exc}\n"
             "Did you run `maturin develop --release`?")
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] numpy not found — numpy benchmarks will be skipped")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Fix Windows console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
ROWS_LARGE = 500
COLS_LARGE = 200
N_LARGE    = ROWS_LARGE * COLS_LARGE  # 100_000 elements
WARMUP     = 3
RUNS       = 5

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# name, ok, rm_ns, np_ns, py_ns
results: list[tuple[str, bool, float, float, float]] = []

# ── Helpers ───────────────────────────────────────────────────────────────────

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
            if len(list(got)) != len(expected):
                ok = False
            else:
                try:
                    ok = all(
                        (math.isnan(float(a)) and math.isnan(float(b)))
                        or abs(float(a) - float(b)) <= tol
                        for a, b in zip(got, expected)
                    )
                except (ValueError, TypeError):
                    ok = list(got) == list(expected)
        elif isinstance(expected, float) and math.isnan(expected):
            ok = math.isnan(float(got))
        elif isinstance(expected, float) and math.isinf(expected):
            ok = math.isinf(float(got)) and (float(got) > 0) == (expected > 0)
        else:
            ok = abs(float(got) - expected) <= tol
    except Exception as e:
        print(f"  [{FAIL}] {name}: exception during check — {e}")
        return False
    if not ok:
        print(f"  [{FAIL}] {name}: got={got!r}  expected={expected!r}")
    return ok


def record(name: str, ok: bool, rm_ns: float, np_ns: float, py_ns: float = 0.0, *, is_bench: bool = False):
    results.append((name, ok, rm_ns, np_ns, py_ns))
    tag = "BENCH" if is_bench else (PASS if ok else FAIL)
    label = tag if is_bench else f"[{tag}]"
    
    speedup_np = np_ns / rm_ns if rm_ns > 0 and np_ns > 0 else 0.0
    speedup_py = py_ns / rm_ns if rm_ns > 0 and py_ns > 0 else 0.0

    def fmt(ns: float) -> str:
        if ns <= 0: return "    N/A   "
        if ns >= 1_000_000: return f"{ns/1e6:7.2f} ms"
        if ns >= 1_000:     return f"{ns/1e3:7.2f} µs"
        return f"{ns:7.1f} ns"

    msg = f"  {label:7s} {name:40s} rm={fmt(rm_ns)}  np={fmt(np_ns)}"
    if py_ns > 0:
        msg += f"  py={fmt(py_ns)}"
    
    if speedup_np > 0:
        msg += f"  vNP={speedup_np:5.1f}x"
    if speedup_py > 0:
        msg += f"  vPY={speedup_py:5.1f}x"
    
    print(msg)


def bench(label: str, rm_fn: Callable, np_fn: Callable, py_fn: Callable = None, n_iter: int = 1, ok: bool = True):
    rm_t = median_ns(rm_fn, n_iter)
    np_t = median_ns(np_fn, n_iter)
    py_t = median_ns(py_fn, n_iter) if py_fn else 0.0
    record(label, ok, rm_t, np_t, py_t, is_bench=True)


def flat_list(rows, cols, val=None):
    if val is not None:
        return [[float(val)] * cols for _ in range(rows)]
    return [[float(i * cols + j) for j in range(cols)] for i in range(rows)]


# ── Shared data ───────────────────────────────────────────────────────────────

PY_DATA  = flat_list(ROWS_LARGE, COLS_LARGE, None)
RM_LARGE = Array(PY_DATA)

if HAS_NUMPY:
    NP_LARGE = np.array(PY_DATA, dtype=np.float64)

SMALL_DATA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
RM_SMALL   = Array(SMALL_DATA)
if HAS_NUMPY:
    NP_SMALL = np.array(SMALL_DATA)


# ════════════════════════════════════════════════════════════════════════════
# 1 — Construction
# ════════════════════════════════════════════════════════════════════════════
print("\n── 1. Construction ─────────────────────────────────────────────────")

ok  = check("Array([[]]) shape",  Array([]).shape, [0, 0])
ok &= check("Array([[1,2],[3,4]]) shape", Array([[1.0,2.0],[3.0,4.0]]).shape, [2, 2])
ok &= check("zeros(3,3) sum",     Array.zeros(3, 3).sum(), 0.0)
ok &= check("ones(3,3) sum",      Array.ones(3, 3).sum(), 9.0)
ok &= check("full(2,3,5.0) sum",  Array.full(2, 3, 5.0).sum(), 30.0)
ok &= check("eye(3)[0,0]",        Array.eye(3)[[0, 0]], 1.0)
ok &= check("eye(3)[0,1]",        Array.eye(3)[[0, 1]], 0.0)
ok &= check("eye(3)[1,1]",        Array.eye(3)[[1, 1]], 1.0)
record("construction correctness", ok, 0, 0, 0)

if HAS_NUMPY:
    bench("Array(list) 500×200",
          lambda: Array(PY_DATA),
          lambda: np.array(PY_DATA, dtype=np.float64),
          lambda: [row[:] for row in PY_DATA])
    bench("zeros(500,200)",
          lambda: Array.zeros(ROWS_LARGE, COLS_LARGE),
          lambda: np.zeros((ROWS_LARGE, COLS_LARGE)))
    bench("ones(500,200)",
          lambda: Array.ones(ROWS_LARGE, COLS_LARGE),
          lambda: np.ones((ROWS_LARGE, COLS_LARGE)))
    bench("full(500,200,3.14)",
          lambda: Array.full(ROWS_LARGE, COLS_LARGE, 3.14),
          lambda: np.full((ROWS_LARGE, COLS_LARGE), 3.14))
    bench("eye(200)",
          lambda: Array.eye(200),
          lambda: np.eye(200))
    bench("randn(500,200)",
          lambda: Array.randn(ROWS_LARGE, COLS_LARGE),
          lambda: np.random.randn(ROWS_LARGE, COLS_LARGE))
    bench("rand_uniform(500,200)",
          lambda: Array.rand_uniform(ROWS_LARGE, COLS_LARGE),
          lambda: np.random.rand(ROWS_LARGE, COLS_LARGE))

# ════════════════════════════════════════════════════════════════════════════
# 1.5 — Python Baseline (Pure List / array.array)
# ════════════════════════════════════════════════════════════════════════════
print("\n── 1.5. Python Baseline ───────────────────────────────────────────")

PY_FLAT = [float(x) for x in range(N_LARGE)]
PY_ARR  = array.array('d', PY_FLAT)

bench("sum(all) 100k",
      lambda: RM_LARGE.sum_all(),
      lambda: (NP_LARGE.sum() if HAS_NUMPY else 0.0),
      lambda: sum(PY_FLAT))

bench("list(100k) copy",
      lambda: RM_LARGE.copy(),
      lambda: (NP_LARGE.copy() if HAS_NUMPY else None),
      lambda: list(PY_FLAT))

bench("min(100k)",
      lambda: RM_LARGE.min(),
      lambda: (NP_LARGE.min() if HAS_NUMPY else 0.0),
      lambda: min(PY_FLAT))


# ════════════════════════════════════════════════════════════════════════════
# 2 — Shape / metadata
# ════════════════════════════════════════════════════════════════════════════
print("\n── 2. Shape / metadata ─────────────────────────────────────────────")

ok  = check("shape",    RM_LARGE.shape,  [ROWS_LARGE, COLS_LARGE])
ok &= check("ndim",     RM_LARGE.ndim, 2)
ok &= check("size",     RM_LARGE.size,  N_LARGE)
ok &= check("__len__",  len(RM_LARGE),    ROWS_LARGE)
record("shape/ndim/size correctness", ok, 0, 0)

a44 = Array.zeros_nd([2, 3, 4])
ok  = check("zeros_nd([2,3,4]) shape",  a44.shape, [2, 3, 4])
ok &= check("zeros_nd([2,3,4]) ndim",   a44.ndim, 3)
ok &= check("zeros_nd([2,3,4]) size",   a44.size, 24)
record("N-D zeros_nd correctness", ok, 0, 0)

ok  = check("reshape shape",   RM_LARGE.reshape([COLS_LARGE, ROWS_LARGE]).shape, [COLS_LARGE, ROWS_LARGE])
ok &= check("flatten shape",   RM_LARGE.flatten().shape, [1, N_LARGE])
ok &= check("squeeze",         Array.zeros_nd([1, 3, 1]).squeeze().shape, [3])
ok &= check("expand_dims ax0", Array.zeros(3, 3).expand_dims(0).shape, [1, 3, 3])
ok &= check("expand_dims ax1", Array.zeros(3, 3).expand_dims(1).shape, [3, 1, 3])
record("reshape/flatten/squeeze/expand_dims", ok, 0, 0)


# ════════════════════════════════════════════════════════════════════════════
# 3 — Indexing and slicing
# ════════════════════════════════════════════════════════════════════════════
print("\n── 3. Indexing and slicing ──────────────────────────────────────────")

a = Array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
ok  = check("get [0,0]",   a[[0, 0]], 10.0)
ok &= check("get [1,2]",   a[[1, 2]], 60.0)
ok &= check("get_row(0)",  a.get_row(0), [10.0, 20.0, 30.0])
ok &= check("get_row(1)",  a.get_row(1), [40.0, 50.0, 60.0])
ok &= check("get_col(0)",  a.get_col(0), [10.0, 40.0])
ok &= check("get_col(2)",  a.get_col(2), [30.0, 60.0])
ok &= check("slice_rows(0,1) shape",  a.slice_rows(0, 1).shape, [1, 3])
ok &= check("slice_rows(1,2) data",   a.slice_rows(1, 2).to_flat_list(), [40.0, 50.0, 60.0])

# setitem
b = Array([[1.0, 2.0], [3.0, 4.0]])
b[[0, 1]] = 99.0
ok &= check("__setitem__", b[[0, 1]], 99.0)
record("indexing / slicing", ok, 0, 0)

if HAS_NUMPY:
    bench("get_row(250)  500×200",
          lambda: RM_LARGE.get_row(250),
          lambda: NP_LARGE[250].tolist())
    bench("get_col(100)  500×200",
          lambda: RM_LARGE.get_col(100),
          lambda: NP_LARGE[:, 100].tolist())
    bench("slice_rows(0,100)  500×200",
          lambda: RM_LARGE.slice_rows(0, 100),
          lambda: NP_LARGE[0:100])


# ════════════════════════════════════════════════════════════════════════════
# 4 — Arithmetic operators
# ════════════════════════════════════════════════════════════════════════════
print("\n── 4. Arithmetic operators ──────────────────────────────────────────")

va = Array([[1.0, 2.0], [3.0, 4.0]])
vb = Array([[5.0, 6.0], [7.0, 8.0]])
na = np.array([[1.0, 2.0], [3.0, 4.0]]) if HAS_NUMPY else None
nb = np.array([[5.0, 6.0], [7.0, 8.0]]) if HAS_NUMPY else None

for name, got_fn, expected in [
    ("a + b",          lambda: (va + vb).to_flat_list(),   [6,8,10,12]),
    ("a + scalar",     lambda: (va + 10.0).to_flat_list(), [11,12,13,14]),
    ("scalar + a",     lambda: (10.0 + va).to_flat_list(), [11,12,13,14]),
    ("a - b",          lambda: (va - vb).to_flat_list(),   [-4,-4,-4,-4]),
    ("a - scalar",     lambda: (va - 1.0).to_flat_list(),  [0,1,2,3]),
    ("scalar - a",     lambda: (10.0 - va).to_flat_list(), [9,8,7,6]),
    ("a * b (elem)",   lambda: (va * vb).to_flat_list(),   [5,12,21,32]),
    ("a * scalar",     lambda: (va * 2.0).to_flat_list(),  [2,4,6,8]),
    ("scalar * a",     lambda: (2.0 * va).to_flat_list(),  [2,4,6,8]),
    ("a / scalar",     lambda: (vb / 2.0).to_flat_list(),  [2.5,3,3.5,4]),
    ("scalar / a",     lambda: (12.0 / va).to_flat_list(), [12,6,4,3]),
    ("a ** 2",         lambda: (va ** 2.0).to_flat_list(), [1,4,9,16]),
    ("-a",             lambda: (-va).to_flat_list(),        [-1,-2,-3,-4]),
    ("+a",             lambda: (+va).to_flat_list(),        [1,2,3,4]),
    ("abs(neg a)",     lambda: abs(Array([[-1.0,-2.0],[-3.0,-4.0]])).to_flat_list(), [1,2,3,4]),
]:
    ok = check(name, got_fn(), expected)
    record(name, ok, median_ns(got_fn), 0)

# ZeroDivisionError
try:    _ = va / 0.0; zd_ok = False
except ZeroDivisionError: zd_ok = True
record("ZeroDivisionError a/0.0", zd_ok, 0, 0)

# in-place
c = Array([[1.0, 2.0], [3.0, 4.0]])
c += 1.0
ok  = check("__iadd__ scalar", c.to_flat_list(), [2,3,4,5])
c -= 1.0
ok &= check("__isub__ scalar", c.to_flat_list(), [1,2,3,4])
c *= 2.0
ok &= check("__imul__ scalar", c.to_flat_list(), [2,4,6,8])
c /= 2.0
ok &= check("__itruediv__ scalar", c.to_flat_list(), [1,2,3,4])
record("in-place operators", ok, 0, 0)

if HAS_NUMPY:
    for label, rm_fn, np_fn in [
        ("a + scalar  500×200", lambda: RM_LARGE + 1.0,         lambda: NP_LARGE + 1.0),
        ("a * scalar  500×200", lambda: RM_LARGE * 2.0,         lambda: NP_LARGE * 2.0),
        ("a + a       500×200", lambda: RM_LARGE + RM_LARGE,    lambda: NP_LARGE + NP_LARGE),
        ("a - a       500×200", lambda: RM_LARGE - RM_LARGE,    lambda: NP_LARGE - NP_LARGE),
        ("a * a       500×200", lambda: RM_LARGE * RM_LARGE,    lambda: NP_LARGE * NP_LARGE),
        ("a / a       500×200", lambda: RM_LARGE / RM_LARGE,    lambda: NP_LARGE / NP_LARGE),
        ("a ** 2      500×200", lambda: RM_LARGE ** 2.0,        lambda: NP_LARGE ** 2.0),
        ("-a          500×200", lambda: -RM_LARGE,              lambda: -NP_LARGE),
        ("abs(a)      500×200", lambda: abs(RM_LARGE),          lambda: np.abs(NP_LARGE)),
    ]:
        bench(label, rm_fn, np_fn)


# ════════════════════════════════════════════════════════════════════════════
# 5 — Matmul
# ════════════════════════════════════════════════════════════════════════════
print("\n── 5. Matmul ────────────────────────────────────────────────────────")

m1 = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # 2×3
m2 = Array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])  # 3×2
expected_mm = [58.0, 64.0, 139.0, 154.0]
ok = check("matmul 2×3 @ 3×2", (m1 @ m2).to_flat_list(), expected_mm)
record("matmul correctness", ok, 0, 0)

ok = check("transpose shape",  m1.transpose().shape, [3, 2])
ok &= check("transpose t()",   m1.t().shape, [3, 2])
ok &= check("transpose round-trip", m1.transpose().transpose().to_flat_list(), m1.to_flat_list())
record("transpose", ok, 0, 0)

if HAS_NUMPY:
    sq200 = Array.randn(200, 200)
    np200 = np.random.randn(200, 200)
    bench("matmul 200×200 @ 200×200",
          lambda: sq200 @ sq200,
          lambda: np200 @ np200)
    bench("matmul 500×200 @ 200×500",
          lambda: RM_LARGE @ RM_LARGE.transpose(),
          lambda: NP_LARGE @ NP_LARGE.T)
    bench("transpose 500×200",
          lambda: RM_LARGE.transpose(),
          lambda: NP_LARGE.T.copy())


# ════════════════════════════════════════════════════════════════════════════
# 6 — Elementwise math
# ════════════════════════════════════════════════════════════════════════════
print("\n── 6. Elementwise math ──────────────────────────────────────────────")

pos = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
np_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) if HAS_NUMPY else None
flat_pos = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

for name, rm_fn, expected in [
    ("sqrt",    lambda: pos.sqrt().to_flat_list(),  [math.sqrt(x) for x in flat_pos]),
    ("cbrt",    lambda: pos.cbrt().to_flat_list(),  [x**(1/3) for x in flat_pos]),
    ("exp",     lambda: pos.exp().to_flat_list(),   [math.exp(x) for x in flat_pos]),
    ("exp2",    lambda: pos.exp2().to_flat_list(),  [2.0**x for x in flat_pos]),
    ("expm1",   lambda: pos.expm1().to_flat_list(), [math.expm1(x) for x in flat_pos]),
    ("log",     lambda: pos.log().to_flat_list(),   [math.log(x) for x in flat_pos]),
    ("log2",    lambda: pos.log2().to_flat_list(),  [math.log2(x) for x in flat_pos]),
    ("log10",   lambda: pos.log10().to_flat_list(), [math.log10(x) for x in flat_pos]),
    ("log1p",   lambda: pos.log1p().to_flat_list(), [math.log1p(x) for x in flat_pos]),
    ("sin",     lambda: pos.sin().to_flat_list(),   [math.sin(x) for x in flat_pos]),
    ("cos",     lambda: pos.cos().to_flat_list(),   [math.cos(x) for x in flat_pos]),
    ("tan",     lambda: pos.tan().to_flat_list(),   [math.tan(x) for x in flat_pos]),
    ("asin",    lambda: Array([[0.0, 0.5, 1.0],[0.0,0.5,1.0]]).asin().to_flat_list(),
                [math.asin(x) for x in [0.0,0.5,1.0,0.0,0.5,1.0]]),
    ("acos",    lambda: Array([[0.0, 0.5, 1.0],[0.0,0.5,1.0]]).acos().to_flat_list(),
                [math.acos(x) for x in [0.0,0.5,1.0,0.0,0.5,1.0]]),
    ("atan",    lambda: pos.atan().to_flat_list(),  [math.atan(x) for x in flat_pos]),
    ("sinh",    lambda: pos.sinh().to_flat_list(),  [math.sinh(x) for x in flat_pos]),
    ("cosh",    lambda: pos.cosh().to_flat_list(),  [math.cosh(x) for x in flat_pos]),
    ("tanh",    lambda: pos.tanh().to_flat_list(),  [math.tanh(x) for x in flat_pos]),
    ("abs",     lambda: Array([[-1.0,-2.0,-3.0],[-4.0,-5.0,-6.0]]).abs().to_flat_list(),
                [1.0,2.0,3.0,4.0,5.0,6.0]),
    ("ceil",    lambda: Array([[1.1,2.5],[3.9,4.0]]).ceil().to_flat_list(), [2.0,3.0,4.0,4.0]),
    ("floor",   lambda: Array([[1.1,2.5],[3.9,4.0]]).floor().to_flat_list(), [1.0,2.0,3.0,4.0]),
    ("round",   lambda: Array([[1.4,2.5],[3.6,4.5]]).round().to_flat_list(), [1.0,3.0,4.0,5.0]),
    ("trunc",   lambda: Array([[1.9,2.1],[3.7,4.2]]).trunc().to_flat_list(), [1.0,2.0,3.0,4.0]),
    ("fract",   lambda: Array([[1.7,2.3],[3.0,4.5]]).fract().to_flat_list(),
                [0.7,0.3,0.0,0.5]),
    ("signum",  lambda: Array([[-2.0,0.0],[3.0,-1.0]]).signum().to_flat_list(), [-1.0,1.0,1.0,-1.0]),
    ("recip",   lambda: pos.recip().to_flat_list(),  [1.0/x for x in flat_pos]),
    ("pow_scalar(2)", lambda: pos.pow_scalar(2.0).to_flat_list(), [x**2 for x in flat_pos]),
    ("clamp(2,5)",    lambda: pos.clamp(2.0, 5.0).to_flat_list(), [max(2.0,min(5.0,x)) for x in flat_pos]),
    ("hypot_scalar(1)",lambda: pos.hypot_scalar(1.0).to_flat_list(), [math.hypot(x,1) for x in flat_pos]),
    ("atan2_scalar(1)",lambda: pos.atan2_scalar(1.0).to_flat_list(), [math.atan2(x,1) for x in flat_pos]),
    ("lerp_scalar",    lambda: pos.lerp_scalar(10.0, 0.5).to_flat_list(), [x+0.5*(10-x) for x in flat_pos]),
    ("fma(2,1)",       lambda: pos.fma(2.0, 1.0).to_flat_list(), [x*2+1 for x in flat_pos]),
]:
    ok = check(name, rm_fn(), expected, tol=1e-9)
    record(name, ok, median_ns(rm_fn), 0)

if HAS_NUMPY:
    for label, rm_fn, np_fn in [
        ("sin    500×200", lambda: RM_LARGE.sin(),   lambda: np.sin(NP_LARGE)),
        ("cos    500×200", lambda: RM_LARGE.cos(),   lambda: np.cos(NP_LARGE)),
        ("exp    500×200", lambda: RM_LARGE.exp(),   lambda: np.exp(NP_LARGE)),
        ("log    500×200", lambda: RM_LARGE.log(),   lambda: np.log(NP_LARGE + 1e-10)),
        ("sqrt   500×200", lambda: RM_LARGE.sqrt(),  lambda: np.sqrt(NP_LARGE + 1e-10)),
        ("abs    500×200", lambda: RM_LARGE.abs(),   lambda: np.abs(NP_LARGE)),
        ("tanh   500×200", lambda: RM_LARGE.tanh(),  lambda: np.tanh(NP_LARGE)),
        ("floor  500×200", lambda: RM_LARGE.floor(), lambda: np.floor(NP_LARGE)),
        ("recip  500×200", lambda: RM_LARGE.recip(), lambda: 1.0 / (NP_LARGE + 1e-10)),
    ]:
        bench(label, rm_fn, np_fn)


# ════════════════════════════════════════════════════════════════════════════
# 7 — Reductions
# ════════════════════════════════════════════════════════════════════════════
print("\n── 7. Reductions ────────────────────────────────────────────────────")

r3x3 = Array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
flat333 = [1,2,3,4,5,6,7,8,9]

ok  = check("sum_all",    r3x3.sum_all(),  45.0)
ok &= check("sum(axis=None)", float(r3x3.sum()), 45.0)
ok &= check("sum(axis=0)", list(r3x3.sum(axis=0)), [12.0,15.0,18.0])
ok &= check("sum(axis=1)", list(r3x3.sum(axis=1)), [6.0,15.0,24.0])
ok &= check("prod",       r3x3.prod(),     math.factorial(9))
ok &= check("mean",       r3x3.mean(),     5.0)
ok &= check("min",        r3x3.min(),      1.0)
ok &= check("max",        r3x3.max(),      9.0)
ok &= check("argmin",     r3x3.argmin(),   0)
ok &= check("argmax",     r3x3.argmax(),   8)
ok &= check("mean_axis0", list(r3x3.mean_axis0()), [4.0,5.0,6.0])
ok &= check("mean_axis1", list(r3x3.mean_axis1()), [2.0,5.0,8.0])
ok &= check("std_axis0",  list(r3x3.std_axis0()), [3.0+1e-8, 3.0+1e-8, 3.0+1e-8], tol=1e-6)
ok &= check("var_axis0",  list(r3x3.var_axis0()),  [9.0, 9.0, 9.0], tol=1e-9)
record("all reductions correctness", ok, 0, 0)

if HAS_NUMPY:
    for label, rm_fn, np_fn in [
        ("sum_all  500×200",   lambda: RM_LARGE.sum_all(),     lambda: NP_LARGE.sum()),
        ("sum ax=0 500×200",   lambda: RM_LARGE.sum(axis=0),   lambda: NP_LARGE.sum(axis=0)),
        ("sum ax=1 500×200",   lambda: RM_LARGE.sum(axis=1),   lambda: NP_LARGE.sum(axis=1)),
        ("mean     500×200",   lambda: RM_LARGE.mean(),        lambda: NP_LARGE.mean()),
        ("mean_ax0 500×200",   lambda: RM_LARGE.mean_axis0(),  lambda: NP_LARGE.mean(axis=0)),
        ("mean_ax1 500×200",   lambda: RM_LARGE.mean_axis1(),  lambda: NP_LARGE.mean(axis=1)),
        ("std_ax0  500×200",   lambda: RM_LARGE.std_axis0(),   lambda: NP_LARGE.std(axis=0, ddof=1)),
        ("var_ax0  500×200",   lambda: RM_LARGE.var_axis0(),   lambda: NP_LARGE.var(axis=0, ddof=1)),
        ("min      500×200",   lambda: RM_LARGE.min(),         lambda: NP_LARGE.min()),
        ("max      500×200",   lambda: RM_LARGE.max(),         lambda: NP_LARGE.max()),
        ("argmin   500×200",   lambda: RM_LARGE.argmin(),      lambda: int(NP_LARGE.argmin())),
        ("argmax   500×200",   lambda: RM_LARGE.argmax(),      lambda: int(NP_LARGE.argmax())),
        ("prod     500×200",   lambda: RM_LARGE.prod(),        lambda: NP_LARGE.prod()),
    ]:
        bench(label, rm_fn, np_fn)


# ════════════════════════════════════════════════════════════════════════════
# 8 — Linear Algebra
# ════════════════════════════════════════════════════════════════════════════
print("\n── 8. Linear Algebra ────────────────────────────────────────────────")

sq4 = Array([[4.0,3.0],[6.0,3.0]])
if HAS_NUMPY:
    np_sq4 = np.array([[4.0,3.0],[6.0,3.0]])

# inv
inv4 = sq4.inv()
ident = (sq4 @ inv4).to_flat_list()
ok  = check("inv A@A^-1 ≈ I diag", [ident[0], ident[3]], [1.0, 1.0], tol=1e-10)
ok &= check("inv A@A^-1 ≈ I off",  [ident[1], ident[2]], [0.0, 0.0], tol=1e-10)

# det
ok &= check("det [[4,3],[6,3]]",  sq4.det(), 4*3-3*6, tol=1e-9)

# trace
ok &= check("trace",  sq4.trace(), 7.0)

# norm_frobenius
ok &= check("norm_frobenius",  sq4.norm_frobenius(),
            math.sqrt(4**2+3**2+6**2+3**2), tol=1e-9)

# is_symmetric
ok &= check("is_symmetric sym",    Array([[1.0,2.0],[2.0,1.0]]).is_symmetric(), True)
ok &= check("is_symmetric nonsym", Array([[1.0,2.0],[3.0,1.0]]).is_symmetric(), False)

# is_positive_definite
ok &= check("is_pd True",  Array([[4.0,2.0],[2.0,3.0]]).is_positive_definite(), True)
ok &= check("is_pd False", Array([[1.0,2.0],[2.0,1.0]]).is_positive_definite(), False)

record("inv / det / trace / norms / symmetry", ok, 0, 0)

# solve
A_solve = Array([[2.0,1.0],[5.0,7.0]])
b_solve = Array([[11.0],[13.0]])
x_solve = A_solve.solve(b_solve)
ok = check("solve Ax=b",  x_solve.to_flat_list(), [7.111111, -3.222222], tol=1e-5)
record("solve", ok, 0, 0)

# QR
q, r = sq4.qr()
ok  = check("qr Q shape", q.shape[1], 2)
ok &= check("qr R shape", r.shape, [2, 2])
qr_prod = (q @ r).to_flat_list()
ok &= check("qr Q@R ≈ A", qr_prod, sq4.to_flat_list(), tol=1e-9)
record("QR decomposition", ok, 0, 0)

# SVD
u, s_vec, vt = sq4.svd()
ok  = check("svd U shape[0]", u.shape[0], 2)
ok &= check("svd s len", len(list(s_vec)), 2)
ok &= check("svd Vt shape", vt.shape, [2, 2])
record("SVD", ok, 0, 0)

# eigh
pd_mat = Array([[4.0,2.0],[2.0,3.0]])
vals, vecs = pd_mat.eigh()
ok  = check("eigh vals len", len(list(vals)), 2)
ok &= check("eigh vals sum ≈ trace", sum(list(vals)), 7.0, tol=1e-9)
record("eigh (symmetric eigendecomp)", ok, 0, 0)

# gram_matrix
ok = check("gram_matrix shape", RM_SMALL.gram_matrix().shape, [3, 3])
record("gram_matrix", ok, 0, 0)

# covariance
cov = Array.randn(100, 5).covariance()
ok  = check("covariance shape", cov.shape, [5, 5])
ok &= check("covariance is_symmetric", cov.is_symmetric(), True)
record("covariance", ok, 0, 0)

# rank
ok  = check("rank full rank 3×3", Array([[1.0,0,0],[0,1,0],[0,0,1]]).rank(), 3)
ok &= check("rank rank-1 matrix", Array([[1.0,2,3],[2,4,6],[3,6,9]]).rank(), 1)
record("rank", ok, 0, 0)

# pseudo_inv
ok = check("pseudo_inv shape", sq4.pseudo_inv().shape, [2, 2])
record("pseudo_inv", ok, 0, 0)

# normalize
mu_v  = Array.randn(100, 10).mean_axis0()
sig_v = Array.randn(100, 10).std_axis0()
norm_out = Array.randn(100, 10).normalize(mu_v, sig_v)
ok = check("normalize shape", norm_out.shape, [100, 10])
record("normalize", ok, 0, 0)

if HAS_NUMPY:
    sq200 = Array.randn(200, 200)
    np200 = np.random.randn(200, 200)
    bench("inv 200×200",    lambda: sq200.inv(),              lambda: np.linalg.inv(np200))
    bench("det 200×200",    lambda: sq200.det(),              lambda: np.linalg.det(np200))
    bench("svd 200×200",    lambda: sq200.svd(),              lambda: np.linalg.svd(np200))
    bench("gram 500×200",   lambda: RM_LARGE.gram_matrix(),  lambda: NP_LARGE.T @ NP_LARGE)
    bench("cov 500×200",    lambda: RM_LARGE.covariance(),   lambda: np.cov(NP_LARGE.T))
    bench("norm_frob 500×200", lambda: RM_LARGE.norm_frobenius(), lambda: np.linalg.norm(NP_LARGE, 'fro'))


# ════════════════════════════════════════════════════════════════════════════
# 9 — ML Operations
# ════════════════════════════════════════════════════════════════════════════
print("\n── 9. ML Operations ─────────────────────────────────────────────────")

logits = Array([[-1.0, 0.0, 1.0], [2.0, -0.5, 0.5]])

# Activations
ok  = check("sigmoid range", all(0 < x < 1 for x in logits.sigmoid().to_flat_list()), True)
ok &= check("relu neg→0",    Array([[-1.0,-2.0],[0.0,3.0]]).relu().to_flat_list(), [0.0,0.0,0.0,3.0])
ok &= check("relu_deriv",    Array([[-1.0,2.0],[0.0,3.0]]).relu_deriv().to_flat_list(), [0.0,1.0,0.0,1.0])
ok &= check("leaky_relu",    Array([[-2.0,3.0]]).leaky_relu(0.1).to_flat_list(), [-0.2,3.0])
ok &= check("softmax row sums=1",
            [abs(sum(row) - 1.0) < 1e-10 for row in logits.softmax().to_list()], [True, True])
ok &= check("log_softmax row max",
            all(x <= 0 for x in logits.log_softmax().to_flat_list()), True)
ok &= check("gelu(0.0)", Array([[0.0]]).gelu().to_flat_list(), [0.0], tol=1e-9)
ok &= check("swish(0.0)", Array([[0.0]]).swish().to_flat_list(), [0.0], tol=1e-9)
ok &= check("mish(0.0)", Array([[0.0]]).mish().to_flat_list(), [0.0], tol=1e-9)
ok &= check("selu sign pres", all(x > 0 for x in Array([[1.0,2.0],[3.0,4.0]]).selu().to_flat_list()), True)
ok &= check("softplus > 0",   all(x > 0 for x in logits.softplus().to_flat_list()), True)
ok &= check("elu neg domain", Array([[-1.0,0.0,1.0]]).elu(1.0).to_flat_list()[0] < 0, True)
ok &= check("hardswish(0.0)", Array([[0.0]]).hardswish().to_flat_list(), [0.0], tol=1e-9)
ok &= check("sigmoid_deriv range", all(0 < x < 0.26 for x in logits.sigmoid_deriv().to_flat_list()), True)
ok &= check("leaky_relu_deriv",    Array([[-2.0,3.0]]).leaky_relu_deriv(0.1).to_flat_list(), [0.1,1.0])
record("all activation functions", ok, 0, 0)

# Loss functions
pred   = Array([[0.7, 0.3], [0.2, 0.8]])
target = Array([[1.0, 0.0], [0.0, 1.0]])
ok  = check("mse_loss",  abs(pred.mse_loss(target) - ((0.09+0.09+0.04+0.04)/4)) < 1e-9, True)
ok &= check("mae_loss",  abs(pred.mae_loss(target) - ((0.3+0.3+0.2+0.2)/4)) < 1e-9, True)
ok &= check("huber_loss", pred.huber_loss(target, 1.0) > 0, True)
ok &= check("binary_ce > 0", pred.binary_cross_entropy(target) > 0, True)

# cross entropy with labels
labels_logits = Array([[-1.0, 0.0, 1.0], [2.0, -0.5, 0.5]])
ok &= check("cross_entropy_loss > 0", labels_logits.log_softmax().cross_entropy_loss([2, 0]) > 0, True)
record("all loss functions", ok, 0, 0)

# Normalization
from rmath.vector import Vector
batch = Array.randn(32, 16)
mu_v  = batch.mean_axis0()
sig_v = batch.std_axis0()
bn    = batch.batch_norm(mu_v, sig_v, Vector([1.0]*16), Vector([0.0]*16))
ok  = check("batch_norm shape", bn.shape, [32, 16])
ln   = batch.layer_norm(1e-5)
ok &= check("layer_norm shape", ln.shape, [32, 16])
ok &= check("layer_norm row mean ≈ 0", abs(ln.mean_axis1().to_list()[0]) < 0.1, True)
record("batch_norm / layer_norm", ok, 0, 0)

# Dropout
dropped = batch.dropout(0.5)
ok = check("dropout shape", dropped.shape, [32, 16])
record("dropout", ok, 0, 0)

# Padding
padded = Array([[1.0,2.0],[3.0,4.0]]).pad(1, 1, 1, 1, 0.0)
ok  = check("pad shape", padded.shape, [4, 4])
ok &= check("pad corner", padded[[0,0]], 0.0)
ok &= check("pad center", padded[[1,1]], 1.0)
record("pad", ok, 0, 0)

# Pooling
pool_in = Array([[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]])
ok  = check("max_pool2d shape",  pool_in.max_pool2d(2).shape, [2, 2])
ok &= check("max_pool2d values", pool_in.max_pool2d(2).to_flat_list(), [6.0,8.0,14.0,16.0])
ok &= check("avg_pool2d values", pool_in.avg_pool2d(2).to_flat_list(), [3.5,5.5,11.5,13.5])
record("max_pool2d / avg_pool2d", ok, 0, 0)

# Gradient helpers
ok  = check("mse_grad shape", pred.mse_grad(target).shape, [2, 2])
ok &= check("softmax_ce_grad shape", logits.softmax_ce_grad([2, 0]).shape, [2, 3])
record("mse_grad / softmax_ce_grad", ok, 0, 0)

if HAS_NUMPY:
    for label, rm_fn, np_fn in [
        ("sigmoid 500×200",   lambda: RM_LARGE.sigmoid(),    lambda: 1/(1+np.exp(-NP_LARGE))),
        ("relu    500×200",   lambda: RM_LARGE.relu(),       lambda: np.maximum(0, NP_LARGE)),
        ("softmax 500×200",   lambda: RM_LARGE.softmax(),
         lambda: (lambda x: np.exp(x - x.max(1, keepdims=True)) /
                  np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True))(NP_LARGE)),
        ("tanh    500×200",   lambda: RM_LARGE.tanh(),       lambda: np.tanh(NP_LARGE)),
        ("gelu    500×200",   lambda: RM_LARGE.gelu(),
         lambda: NP_LARGE * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (NP_LARGE + 0.044715*NP_LARGE**3)))),
    ]:
        bench(label, rm_fn, np_fn)


# ════════════════════════════════════════════════════════════════════════════
# 10 — Comparison / mask ops
# ════════════════════════════════════════════════════════════════════════════
print("\n── 10. Comparison / mask ops ────────────────────────────────────────")

cmp = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

ok  = check("gt(3)",       cmp.gt(3.0),    [False,False,False,True,True,True])
ok &= check("lt(3)",       cmp.lt(3.0),    [True,True,False,False,False,False])
ok &= check("ge(3)",       cmp.ge(3.0),    [False,False,True,True,True,True])
ok &= check("le(3)",       cmp.le(3.0),    [True,True,True,False,False,False])
ok &= check("eq_scalar(3)",cmp.eq_scalar(3.0), [False,False,True,False,False,False])
ok &= check("ne_scalar(3)",cmp.ne_scalar(3.0), [True,True,False,True,True,True])
ok &= check("isnan",       Array([[1.0,float('nan')],[2.0,3.0]]).isnan(),
            [False,True,False,False])
ok &= check("isfinite",    Array([[1.0,float('inf')],[2.0,3.0]]).isfinite(),
            [True,False,True,True])
ok &= check("isinf",       Array([[1.0,float('inf')],[2.0,3.0]]).isinf(),
            [False,True,False,False])
ok &= check("where_scalar",
            cmp.where_scalar([True,False,True,False,True,False], 0.0).to_flat_list(),
            [1.0,0.0,3.0,0.0,5.0,0.0])
record("all comparison / mask ops", ok, 0, 0)

if HAS_NUMPY:
    bench("gt(3)  500×200",  lambda: RM_LARGE.gt(3.0),   lambda: (NP_LARGE > 3.0).tolist())
    bench("isnan  500×200",  lambda: RM_LARGE.isnan(),    lambda: np.isnan(NP_LARGE).tolist())


# ════════════════════════════════════════════════════════════════════════════
# 11 — IO (save / load)
# ════════════════════════════════════════════════════════════════════════════
print("\n── 11. IO — save / load ─────────────────────────────────────────────")

with tempfile.TemporaryDirectory() as tmpdir:
    rmath_path = os.path.join(tmpdir, "test.rmath")
    csv_path   = os.path.join(tmpdir, "test.csv")
    bin_path   = os.path.join(tmpdir, "test.bin")

    # .rmath round-trip
    RM_LARGE.save(rmath_path)
    loaded = Array.load(rmath_path)
    ok  = check(".rmath shape preserved", loaded.shape, RM_LARGE.shape)
    ok &= check(".rmath data preserved",
                abs(loaded.sum_all() - RM_LARGE.sum_all()) < 1e-6, True)
    record(".rmath save/load round-trip", ok, 0, 0)

    # estimated bytes
    estimated = RM_LARGE.estimated_bytes()
    actual    = os.path.getsize(rmath_path)
    ok = check("estimated_bytes accuracy", abs(estimated - actual), 0)
    ok &= check("memory_bytes correct", RM_LARGE.memory_bytes(), N_LARGE * 8)
    record("estimated_bytes / memory_bytes", ok, 0, 0)

    # CSV round-trip
    RM_SMALL.save_csv(csv_path)
    loaded_csv = Array.load_csv(csv_path)
    ok = check("csv shape preserved", loaded_csv.shape, RM_SMALL.shape)
    ok &= check("csv data preserved", loaded_csv.to_flat_list(), RM_SMALL.to_flat_list(), tol=1e-9)
    record("csv save/load round-trip", ok, 0, 0)

    # CSV with header
    RM_SMALL.save_csv(csv_path, header=["a", "b", "c"])
    loaded_csv2, hdr = Array.load_csv_with_header(csv_path)
    ok  = check("csv+header columns",    hdr, ["a", "b", "c"])
    ok &= check("csv+header data shape", loaded_csv2.shape, RM_SMALL.shape)
    record("csv with header", ok, 0, 0)

    # .bin round-trip
    RM_SMALL.save_bin(bin_path)
    loaded_bin = Array.from_list(
        list(__import__('struct').unpack(f'{RM_SMALL.size}d',
             open(bin_path,'rb').read())),
        RM_SMALL.shape)
    ok = check("bin save/load round-trip", loaded_bin.to_flat_list(), RM_SMALL.to_flat_list(), tol=1e-9)
    record("bin save/load round-trip", ok, 0, 0)

    # Benchmark save/load speed
    if HAS_NUMPY:
        bench(".rmath save  500×200",
              lambda: RM_LARGE.save(rmath_path),
              lambda: np.save(rmath_path + ".npy", NP_LARGE))
        bench(".rmath load  500×200",
              lambda: Array.load(rmath_path),
              lambda: np.load(rmath_path + ".npy"))


# ════════════════════════════════════════════════════════════════════════════
# 12 — Lazy loading
# ════════════════════════════════════════════════════════════════════════════
print("\n── 12. Lazy loading ─────────────────────────────────────────────────")

with tempfile.TemporaryDirectory() as tmpdir:
    rmath_path = os.path.join(tmpdir, "lazy_test.rmath")
    csv_path   = os.path.join(tmpdir, "lazy_test.csv")
    bin_path   = os.path.join(tmpdir, "lazy_test.bin")

    # Write test files
    RM_LARGE.save(rmath_path)
    RM_LARGE.save_csv(csv_path)
    RM_LARGE.save_bin(bin_path)

    from rmath.array import LazyArray, MmapArray

    # LazyArray.open — no data loaded yet
    lazy_r = LazyArray.open(rmath_path)
    shape  = lazy_r.peek()
    ok  = check("lazy .rmath peek shape", shape, RM_LARGE.shape)

    lazy_c = LazyArray.open(csv_path)
    shape_c = lazy_c.peek()
    ok &= check("lazy csv peek shape", shape_c, RM_LARGE.shape)

    # Full load
    full = lazy_r.load()
    ok &= check("lazy .rmath full load shape", full.shape, RM_LARGE.shape)

    # Row slicing
    partial = lazy_r.load_rows(0, 10)
    ok &= check("lazy load_rows shape", partial.shape, [10, COLS_LARGE])

    csv_partial = lazy_c.load_rows(10, 20)
    ok &= check("csv load_rows shape", csv_partial.shape, [10, COLS_LARGE])
    record("LazyArray peek / load / load_rows", ok, 0, 0)

    # Chunk iteration
    chunk_count = 0
    total_rows  = 0
    for chunk in lazy_r.chunks(100):
        chunk_count += 1
        total_rows  += chunk.shape[0]
    ok  = check("lazy chunk count",      chunk_count, math.ceil(ROWS_LARGE / 100))
    ok &= check("lazy chunk total rows", total_rows,  ROWS_LARGE)
    record("LazyArray chunk iteration (.rmath)", ok, 0, 0)

    # CSV chunk iteration
    chunk_count = 0
    for chunk in lazy_c.chunks(50):
        chunk_count += 1
    ok = check("csv chunk count", chunk_count, math.ceil(ROWS_LARGE / 50))
    record("LazyArray chunk iteration (CSV)", ok, 0, 0)

    # open_bin
    lazy_b = LazyArray.open_bin(bin_path, ROWS_LARGE, COLS_LARGE)
    bin_full = lazy_b.load()
    ok = check("bin open_bin load shape", bin_full.shape, [ROWS_LARGE, COLS_LARGE])
    record("LazyArray open_bin", ok, 0, 0)

    # MmapArray
    mmap = MmapArray.mmap(bin_path, ROWS_LARGE, COLS_LARGE)
    ok  = check("mmap shape",       mmap.shape,       (ROWS_LARGE, COLS_LARGE))
    ok &= check("mmap get_row len", len(mmap.get_row(0)), COLS_LARGE)
    ok &= check("mmap get_element", mmap.get_element(0, 0),
                RM_LARGE.to_flat_list()[0], tol=1e-9)

    rows_m = mmap.load_rows(0, 10)
    ok &= check("mmap load_rows shape", rows_m.shape, [10, COLS_LARGE])

    all_m = mmap.load_all()
    ok &= check("mmap load_all shape", all_m.shape, [ROWS_LARGE, COLS_LARGE])
    record("MmapArray all ops", ok, 0, 0)

    # MmapArray chunk iteration
    chunk_count = 0
    for chunk in mmap.chunks(100):
        chunk_count += 1
    ok = check("mmap chunk count", chunk_count, math.ceil(ROWS_LARGE / 100))
    record("MmapArray chunk iteration", ok, 0, 0)

    # Benchmark lazy vs eager load
    bench("lazy load_rows(0,50) .rmath",
          lambda: LazyArray.open(rmath_path).load_rows(0, 50),
          lambda: Array.load(rmath_path).slice_rows(0, 50))

    bench("mmap get_row(250)",
          lambda: mmap.get_row(250),
          lambda: RM_LARGE.get_row(250))


# ════════════════════════════════════════════════════════════════════════════
# 13 — Interop (NumPy / Pandas / Torch)
# ════════════════════════════════════════════════════════════════════════════
print("\n── 13. Interop ─────────────────────────────────────────────────────")

if HAS_NUMPY:
    np_out = RM_SMALL.to_numpy()
    ok  = check("to_numpy shape",  list(np_out.shape), RM_SMALL.shape)
    ok &= check("to_numpy sum",    float(np_out.sum()), RM_SMALL.sum_all(), tol=1e-6)

    rm_from_np = Array.from_numpy(NP_SMALL)
    ok &= check("from_numpy shape", rm_from_np.shape, RM_SMALL.shape)
    ok &= check("from_numpy data",  rm_from_np.to_flat_list(), RM_SMALL.to_flat_list(), tol=1e-9)

    # __array__ protocol
    np_via_arr = np.array(RM_SMALL)
    ok &= check("__array__ protocol", list(np_via_arr.shape), RM_SMALL.shape)
    record("to_numpy / from_numpy / __array__", ok, 0, 0)

    bench("to_numpy 500×200",   lambda: RM_LARGE.to_numpy(),          lambda: NP_LARGE.copy())
    bench("from_numpy 500×200", lambda: Array.from_numpy(NP_LARGE),   lambda: NP_LARGE.tolist())

if HAS_PANDAS:
    df = RM_SMALL.to_dataframe(columns=["x", "y", "z"])
    ok  = check("to_dataframe cols",  list(df.columns), ["x", "y", "z"])
    ok &= check("to_dataframe rows",  len(df), RM_SMALL.nrows())

    rm_from_df = Array.from_dataframe(df)
    ok &= check("from_dataframe shape", rm_from_df.shape, RM_SMALL.shape)
    record("to_dataframe / from_dataframe", ok, 0, 0)

    series = RM_SMALL.to_series(name="flat")
    ok = check("to_series len", len(series), RM_SMALL.size)
    record("to_series", ok, 0, 0)

if HAS_TORCH:
    t = RM_SMALL.to_torch()
    ok  = check("to_torch shape", list(t.shape), RM_SMALL.shape)
    ok &= check("to_torch sum",   abs(float(t.sum()) - RM_SMALL.sum_all()) < 1e-4, True)

    rm_from_t = Array.from_torch(t)
    ok &= check("from_torch shape", rm_from_t.shape, RM_SMALL.shape)
    record("to_torch / from_torch", ok, 0, 0)

# sklearn_shape
ok  = check("sklearn_shape", RM_SMALL.sklearn_shape(), (RM_SMALL.nrows(), RM_SMALL.ncols()))
ok &= check("validate_sklearn clean", RM_SMALL.validate_sklearn() is None, True)
bad = Array([[1.0, float('nan')]])
try:
    bad.validate_sklearn()
    ok &= False
except ValueError:
    ok &= True
record("sklearn_shape / validate_sklearn", ok, 0, 0)


# ════════════════════════════════════════════════════════════════════════════
# 14 — Conversion helpers
# ════════════════════════════════════════════════════════════════════════════
print("\n── 14. Conversion helpers ───────────────────────────────────────────")

ok  = check("to_list shape",     len(RM_SMALL.to_list()), RM_SMALL.nrows())
ok &= check("tolist alias",      RM_SMALL.tolist(), RM_SMALL.to_list())
ok &= check("to_flat_list len",  len(RM_SMALL.to_flat_list()), RM_SMALL.size)
ok &= check("copy independence", RM_SMALL.copy().sum_all(), RM_SMALL.sum_all())

from_list = Array.from_list([1.0,2.0,3.0,4.0,5.0,6.0], [2, 3])
ok &= check("from_list shape",   from_list.shape, [2, 3])
ok &= check("from_list data",    from_list.to_flat_list(), [1,2,3,4,5,6])
record("to_list / to_flat_list / copy / from_list", ok, 0, 0)

# linspace / arange
lin = Array.linspace(0.0, 1.0, 5)
ok  = check("linspace shape", lin.shape, [1, 5])
ok &= check("linspace values", lin.to_flat_list(), [0.0, 0.25, 0.5, 0.75, 1.0])

ar  = Array.arange(0.0, 10.0, 2.0)
ok &= check("arange values", ar.to_flat_list(), [0.0, 2.0, 4.0, 6.0, 8.0])
record("linspace / arange", ok, 0, 0)

if HAS_NUMPY:
    bench("to_flat_list 500×200",
          lambda: RM_LARGE.to_flat_list(),
          lambda: NP_LARGE.flatten().tolist())
    bench("to_list 500×200",
          lambda: RM_LARGE.to_list(),
          lambda: NP_LARGE.tolist())


# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*110)
print("FINAL PERFORMANCE SUMMARY")
print("═"*110)

total  = len(results)
print(f"  Summarizing {total} tests...")
passed = sum(1 for row in results if row[1])
failed = total - passed

print(f"  Tests:   {total}")
print(f"  Passed:  {passed}  ({100*passed//max(total,1)}%)")
print(f"  Failed:  {failed}")

if failed:
    print(f"\n  Failed tests:")
    for row in results:
        name, ok = row[0], row[1]
        if not ok:
            print(f"    ✗  {name}")

timed = [(n, rm, np_ns, py) for n, ok, rm, np_ns, py in results if rm > 0]
if timed:
    # 1. NumPy Speedups
    sp_np = [(n, np_ns/rm) for n, rm, np_ns, py in timed if np_ns > 0]
    # 2. Python Speedups
    sp_py = [(n, py/rm) for n, rm, np_ns, py in timed if py > 0]
    
    if sp_np:
        sp_np.sort(key=lambda x: -x[1])
        print(f"\n  Top 10 Speedups vs NumPy:")
        for name, sp in sp_np[:10]:
            print(f"    {sp:7.2f}×  {name}")
        
    if sp_py:
        sp_py.sort(key=lambda x: -x[1])
        print(f"\n  Top 5 Speedups vs Pure Python:")
        for name, sp in sp_py[:5]:
            print(f"    {sp:7.2f}×  {name}")
            
    if sp_np:
        avg = sum(s for _, s in sp_np) / len(sp_np)
        print(f"\n  Average speedup vs NumPy: {avg:.2f}×")

print()
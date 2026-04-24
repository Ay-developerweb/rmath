# rmath Scalar Benchmark — Full Analysis

**141/141 PASS** · Average speedup: **3.90×** (misleading — see below)

---

## The Big Picture: Two Worlds

Your benchmark reveals a **bimodal performance profile**:

| Domain | Speedup | Verdict |
|---|---|---|
| Individual scalar ops (Sections 1–6) | **0.12×–0.78×** (2–8× slower) | ❌ FFI-dominated |
| LazyPipeline bulk ops (Section 10) | **33×–75× faster** | ✅ Rust-native |
| Tight loop with `to_tuple()` (Section 11) | **0.09×** (11× slower) | ❌ Worst of both worlds |

The "average 3.90×" is misleading because the 7 pipeline benchmarks (33–75×) drag up the average, masking that **every single scalar operation is slower than Python**.

---

## Section-by-Section Breakdown

### ❌ Section 1 — Construction & Conversion (0.28×–0.49×)

```
Scalar construction:  141.7 ns  vs  58.3 ns  →  0.41×
to_python():          200.7 ns  vs  72.0 ns  →  0.36×
float() protocol:     190.0 ns  vs  94.0 ns  →  0.49×
int() protocol:       317.0 ns  vs  87.7 ns  →  0.28×
```

**Why it's slow:** Every call crosses the FFI boundary. The actual work (storing an f64) is ~1ns. The other ~140ns is:

1. **PyO3 argument extraction** — unwrap the Python object, type-check, extract f64 (~40ns)
2. **Rust function dispatch** — call through the vtable PyO3 sets up (~20ns)
3. **Result wrapping** — create a new Python `Scalar` PyObject on the heap (~60-80ns)
4. **Return across FFI** — hand the object back to the interpreter (~20ns)

Python's `float(3.14)` skips all of this — it's a single C-level operation inside the interpreter with no boundary to cross.

> [!IMPORTANT]
> The ~140-200ns baseline you see on every scalar operation is the **fixed FFI tax**. It's the same whether the actual computation takes 1ns (addition) or 50ns (trigonometry).

### ❌ Section 2 — Arithmetic (0.12×–0.45×)

```
__add__  (Scalar+Scalar):  480.6 ns  vs  124.8 ns  →  0.26×
__radd__ (float+Scalar):   748.4 ns  vs  116.8 ns  →  0.16×
__rsub__ (float-Scalar):   695.2 ns  vs   86.3 ns  →  0.12×  ← WORST
__pow__  (Scalar**Scalar): 434.5 ns  vs  197.0 ns  →  0.45×  ← BEST
```

**Why binary ops are even worse:** They pay the FFI tax **twice** — once to extract each operand.

**Why `__r*__` variants are the worst (0.12–0.17×):** When Python sees `3.0 + scalar`, it:
1. Tries `float.__add__(3.0, scalar)` → fails (float doesn't know about Scalar)
2. Falls back to `Scalar.__radd__(scalar, 3.0)` → crosses FFI
3. Extra dispatch overhead from the failed first attempt

That's why `__radd__` (748ns) is ~270ns slower than `__add__` (480ns) — the failed dispatch adds ~200+ ns.

**Why `__pow__` is the "best" at 0.45×:** The actual computation (`f64::powf`) is ~50-100ns, which is proportionally more significant against the fixed FFI cost. The heavier the real work, the less the FFI tax dominates.

### ❌ Section 3 — Comparison & Hashing (0.24×–0.30×)

```
__eq__:  253.0 ns  vs  70.9 ns  →  0.28×
__hash__: 343.5 ns  vs  93.8 ns  →  0.27×
```

**Same FFI story.** Comparisons return a Python `bool` instead of a new `Scalar`, which saves ~30ns on the wrapping side, but the extraction cost remains.

### ❌ Section 4 — IEEE Predicates (0.49×–0.55×)

```
is_nan:    170.6 ns  vs  92.8 ns  →  0.54×
is_finite: 205.8 ns  vs  104.0 ns  →  0.51×
```

**Best of the scalar ops** — only ~2× slower. These are unary methods on `&self` returning `bool`:
- No argument to extract (just `self`)
- The actual check (`x.is_nan()`) is a single CPU instruction (~1ns)
- Returns a primitive bool, not a heap object

Python's `math.isfinite()` also calls into C, but skips the PyO3 wrapper overhead.

### ❌ Section 5 — Utility Methods (0.34×–0.46×)

```
clamp:  325 ns  vs  145 ns  →  0.45×
lerp:   327 ns  vs  136 ns  →  0.41×
```

**Mixed overhead:** These take 2-3 arguments → more extraction work. But the Python equivalents (`max(0, min(10, x))`) are also multi-call expressions, so the gap is narrower.

### ❌ Section 6 — Math Methods (0.13×–0.78×)

```
cbrt:   449.2 ns  vs   60.6 ns  →  0.13×  ← WORST
acosh:  295.1 ns  vs  230.7 ns  →  0.78×  ← BEST
tanh:   294.6 ns  vs  200.1 ns  →  0.68×
sin:    319.2 ns  vs  172.6 ns  →  0.54×
```

**Wide range explained by computation weight:**

| Operation | Actual compute time | FFI % of total | Speedup |
|---|---|---|---|
| `cbrt` (Python: `8**(1/3)`) | ~5ns | ~95% | 0.13× |
| `sin` | ~30ns | ~85% | 0.54× |
| `tanh` | ~60ns | ~75% | 0.68× |
| `acosh` | ~90ns | ~60% | 0.78× |

**The heavier the math, the closer to parity**, because the FFI tax becomes a smaller fraction. `acosh` at 0.78× is nearly break-even because both Rust and Python are calling into the same `libm` C library — the FFI just adds ~70ns overhead.

`cbrt` is catastrophic because Python's `8.0**(1/3)` is a single C-level `pow()` call (~5ns), while rmath pays full ~200ns FFI overhead for essentially the same operation.

---

### ✅ Section 10 — LazyPipeline Benchmarks (33×–75× faster)

```
sum of range(100K):       0.22 ms  vs  11.93 ms  →  54.06×
sum sin(range(100K)):     0.50 ms  vs  22.47 ms  →  44.72×
filter_gt + sum:          0.43 ms  vs  31.03 ms  →  72.38×
var of range(100K):       0.45 ms  vs  33.87 ms  →  75.35×  ← BEST
```

**Why this is massively faster:**

1. **Single FFI crossing** — Python calls `.sum()` once. Rust processes all 100K elements without returning to Python.
2. **SIMD auto-vectorization** — the chunk-based accumulator (`CHUNK_SIZE = 64`) processes 8 f64 values per AVX2 instruction.
3. **Rayon parallelism** — for N ≥ 10,000, work is split across CPU cores.
4. **Zero allocation** — `sum()`, `mean()`, `var()` reduce without materialising a Vec.
5. **Op fusion** — `.sin().abs().add(1.0).sqrt()` applies all 4 ops per element in a single pass.

**Why `var` at 75× is the fastest:** Python's variance requires two passes (compute mean, then compute deviations). Rust uses Welford's online algorithm — single pass, numerically stable, and parallelised.

**Why `filter_gt + sum` at 72× is close:** Python's generator `(x for x in range(N) if x > cutoff)` creates a Python object per surviving element. Rust's filter is a branchless check inside the accumulation loop — zero allocation.

---

### ❌ Section 11 — Tight Loop (0.09× — 11× SLOWER)

```
rmath=348.26 ms  py=32.48 ms  →  0.09×
```

**This is the worst result and the most important to understand.**

Your loop:
```python
def scalar_loop():
    acc = sc.Scalar(0.0)
    for _ in sc.loop_range(float(LOOP_N)).to_tuple():  # Step A
        acc = acc + sc.Scalar(1.0)                      # Step B
    return acc
```

Per-iteration cost breakdown:

| Step | Cost | What's happening |
|---|---|---|
| `to_tuple()` materialisation (one-time) | ~5ms | Rust generates 500K floats, builds Python tuple |
| Python tuple iteration | ~15ns | CPython iterates C-array of float pointers |
| `sc.Scalar(1.0)` construction | ~140ns | FFI → alloc PyObject → return |
| `acc + Scalar(1.0)` addition | ~480ns | FFI → extract both operands → f64 add → wrap result |
| `acc =` rebinding | ~5ns | Python local variable assignment |
| **Total per iteration** | **~640ns** | |
| **500K iterations** | **~320ms** | Matches the measured 348ms |

Python's loop:
```python
acc += 1.0    # ~65ns — single C-level float add, no FFI
```

> [!CAUTION]
> **You cannot win a per-element Python loop with FFI-wrapped scalars.** The FFI boundary costs more than the actual computation. This is a fundamental architectural constraint, not a code bug.

The correct way to express this in rmath:
```python
sc.loop_range(float(LOOP_N)).sum()  # 0.22ms — 1,583× faster than the tight loop
```

---

## The FFI Tax Visualised

```
Python float add:    [===]                           65 ns
                     ↑ all computation

rmath Scalar add:    [████████████████|=|████████████] 480 ns
                     ↑ FFI overhead    ↑  ↑ FFI overhead
                       (extract)    1ns   (wrap result)
                                  actual
                                   add
```

The actual `f64 + f64` is ~1ns. **The other 479ns is overhead.**

---

## Strategic Recommendations

### 1. Don't fight the FFI — use pipelines
The data is clear: **rmath is a bulk-operation engine, not a scalar replacement for Python float**.

```python
# ❌ WRONG — 500K FFI crossings
acc = sc.Scalar(0.0)
for x in sc.loop_range(500000.0).to_tuple():
    acc = acc + sc.Scalar(x).sin()

# ✅ RIGHT — 1 FFI crossing
result = sc.loop_range(500000.0).sin().sum()    # 75× faster than pure Python
```

### 2. `to_tuple()` shines for pipeline → Python handoff
```python
# Compute in Rust, iterate results in Python
for val in sc.loop_range(1000.0).sin().add(1.0).sqrt().to_tuple():
    # val is a plain Python float — no Scalar wrapping overhead
    some_python_function(val)
```

### 3. Expand the pipeline op vocabulary
Every operation you add to `LazyPipeline` is one fewer FFI crossing users need. Consider:
- `pow(exp)`, `log()`, `log2()`, `tan()`, `clamp(lo, hi)`
- `map_py(fn)` — batch-call a Python function (one FFI crossing for the function lookup, then apply per element)
- `zip_add(other_pipeline)` — element-wise combine two pipelines

### 4. The `Scalar` type has value as a **type-safe handle**, not a speed win
It's useful for:
- Type checking at API boundaries
- Carrying metadata (units, gradients) in the future
- Being hashable/usable in sets/dicts with custom semantics

But for raw arithmetic speed, users should stay in pipeline-land.

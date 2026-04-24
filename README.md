<p align="center">
  <img src="docs/portal/logo.svg" width="150" alt="RMath Logo">
</p>

<h3 align="center">Drop-in numerical accelerator for the Python computing ecosystem.</h3>

<p align="center">
  <a href="https://pypi.org/project/rmath-py/"><img src="https://img.shields.io/pypi/v/rmath-py?color=blue" alt="PyPI"></a>
  <a href="https://github.com/Ay-developerweb/rmath/actions/workflows/publish.yml"><img src="https://github.com/Ay-developerweb/rmath/actions/workflows/publish.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/rmath-py" alt="Python"></a>
</p>

---

RMath is a high-speed accelerator that offloads heavy mathematical workloads to Rust,
seamlessly integrating back into Python via [PyO3](https://pyo3.rs). Array operations, linear
algebra, statistics, calculus, autodiff, and signal processing all execute
**outside the GIL** on a [Rayon](https://github.com/rayon-rs/rayon) thread pool.

> **Drop-in accelerator, not a replacement.**
> Interop with NumPy, PyTorch, JAX, pandas, and scikit-learn via zero-copy NumPy bridges.

## Install

```bash
pip install rmath-py
```

Pre-built wheels are available for Windows, Linux, and macOS.
No Rust toolchain required.

## Architecture

```
rmath
│
├── Core Types
│   ├── Scalar          — atomic f64 math unit
│   ├── Vector          — 1D optimized operations
│   ├── Array           — N-dimensional compute engine
│   ├── Tensor          — autodiff-enabled (forward + reverse mode)
│   ├── Dual            — forward-mode automatic differentiation
│   └── LazyArray       — memory-efficient large-scale data
│
├── Math Domains
│   ├── linalg          — LU, QR, Cholesky, SVD (via faer)
│   ├── stats           — descriptive + inferential statistics
│   ├── geometry        — 3D transforms, quaternions, convex hull
│   ├── signal          — FFT, convolution, spectral analysis
│   └── special         — gamma, beta, error functions
│
├── ML & Autodiff
│   ├── nn              — activations, loss, normalization
│   ├── Tensor          — reverse-mode gradient tracking
│   └── Dual            — forward-mode differentiation
│
├── Utilities
│   ├── constants       — mathematical and physical constants
│   └── loop_range      — lazy pipeline engine
│
└── Interop
    ├── NumPy           — from_numpy / to_numpy
    ├── PyTorch         — from_torch / to_torch
    ├── JAX             — from_jax / to_jax
    └── pandas          — from_dataframe / to_dataframe
```

## Modules

| Module | Description |
|--------|-------------|
| `rmath.array` | N-dimensional array with automatic storage tiering (stack / heap / mmap) |
| `rmath.vector` | 1-D parallel engine — trig, reductions, sorting, filtering, complex numbers |
| `rmath.scalar` | Precision f64 math — 80+ functions mirroring Python's `math` module |
| `rmath.linalg` | Matrix solvers (LU, QR, Cholesky, SVD) via [faer](https://github.com/sarah-ek/faer-rs) |
| `rmath.stats` | Descriptive and inferential statistics — Welford's algorithm, distributions, regression |
| `rmath.calculus` | Automatic differentiation (dual numbers), numerical integration, root-finding |
| `rmath.geometry` | 3D transforms, quaternions, convex hull |
| `rmath.signal` | FFT, convolution, spectral analysis |
| `rmath.nn` | Activation functions (GELU, Softmax), loss, normalization layers |
| `rmath.special` | Gamma, beta, and error functions |
| `rmath.constants` | Mathematical and physical constants |

---

## Domain Examples

### 1. Scientific Computing / Numerical Analysis

**Solving a linear system + residual validation**

```python
import rmath as rm

A = rm.Array([[4.0, 2.0], [1.0, 3.0]])
b = rm.Array([[1.0], [2.0]])

x = rm.linalg.solve(A, b)

# Validate: residual should be ~0
residual = A.matmul(x).sub(b).norm_frobenius()
print("Solution:", x)
print("Residual:", residual)
```

**Eigendecomposition + reconstruction check**

```python
A = rm.Array([[2.0, 1.0], [1.0, 2.0]])

eigvals, eigvecs = rm.linalg.eigh(A)

# Reconstruct: A = V * Lambda * V^T
Lambda = rm.Array.zeros(2, 2)
Lambda[0, 0] = eigvals[0]
Lambda[1, 1] = eigvals[1]

reconstructed = eigvecs.matmul(Lambda).matmul(eigvecs.t())
error = A.sub(reconstructed).norm_frobenius()
print("Reconstruction error:", error)  # ~1e-16
```

---

### 2. Data Science / Data Analysis

**Load, extract columns, correlate**

```python
import rmath as rm

data = rm.Array([[25, 50000], [30, 60000], [22, 45000], [35, 80000]])

ages = rm.Vector(data.get_col(0))
income = rm.Vector(data.get_col(1))

print("Mean age:", ages.mean())
print("Income std:", income.std_dev())

corr = rm.stats.correlation(ages, income)
print("Correlation:", corr)  # 0.98
```

---

### 3. Statistics / Research

**Descriptive statistics**

```python
import rmath as rm

v = rm.Vector([2, 4, 4, 4, 5, 5, 7, 9])

print("Mean:", v.mean())          # 5.0
print("Variance:", v.variance())  # 4.57
print("Std Dev:", v.std_dev())    # 2.14
```

**Hypothesis testing**

```python
group1 = rm.Vector([20, 22, 19, 24, 25])
group2 = rm.Vector([30, 29, 35, 32, 31])

t_stat, p_value = rm.stats.t_test_independent(group1, group2)
print("t-stat:", t_stat)   # -6.12
print("p-value:", p_value) # 0.001
```

**Linear regression**

```python
x = rm.Vector([1, 2, 3, 4, 5])
y = rm.Vector([2, 4, 5, 4, 5])

result = rm.stats.linear_regression(x, y)
print(f"y = {result['slope']}*x + {result['intercept']}")
print(f"R² = {result['r_squared']}")
```

---

### 4. Financial / Economic Analysis

**Return analysis**

```python
import rmath as rm

prices = rm.Vector([100, 102, 101, 105, 110])
diffs = prices.diff()  # [2, -1, 4, 5]
print("Price changes:", list(diffs))
```

**Covariance matrix**

```python
data = rm.Array([
    [0.01, 0.02, 0.015],
    [0.03, 0.01, 0.02],
    [0.02, 0.025, 0.03],
])

cov = data.covariance()
print("Covariance matrix:", cov)  # 3x3
```

---

### 5. Machine Learning (Autograd)

**Gradient computation**

```python
import rmath as rm

x = rm.Tensor([1.0, 2.0, 3.0], requires_grad=True)

y = (x * x).sum()
y.backward()

print("Gradients:", x.grad)  # [2.0, 4.0, 6.0]
```

**Simple neural step**

```python
w = rm.Tensor.randn(3, requires_grad=True)
x = rm.Tensor([1.0, 2.0, 3.0])

y_pred = (w * x).sum()
target = rm.Tensor([10.0])
loss = ((y_pred - target) * (y_pred - target)).sum()
loss.backward()

print("Loss:", loss.data.to_flat_list()[0])
print("Gradients:", w.grad)
```

**Built-in activations**

```python
x = rm.Array([-1.0, 0.0, 2.0])
print(x.relu())  # [0.0, 0.0, 2.0]
```

---

### 6. Calculus / Differentiation

**Forward-mode autodiff (dual numbers)**

```python
import rmath.calculus as rc

x = rc.Dual(2.0, 1.0)  # value=2, seed=1

y = x.sin() * x.exp()

print("f(2)  =", y.value)       # 6.72
print("f'(2) =", y.derivative)  # 3.64
```

**Numerical integration**

```python
import rmath as rm

result = rm.calculus.integrate_simpson(lambda x: x * x, 0, 1, 100)
print("Integral of x² from 0 to 1:", result)  # 0.3333...
```

---

### 7. Signal Processing

**1D Convolution (FFT-accelerated)**

```python
import rmath as rm

signal = rm.Vector([1, 2, 3, 4])
kernel = rm.Vector([1, 0, -1])

filtered = rm.signal.convolve(signal, kernel, "full")
print(list(filtered))  # [1, 2, 2, 2, -3, -4]
```

**FFT**

```python
signal = rm.Vector([1, 0, 0, 0])

fft_result = rm.signal.fft(signal)
print("Magnitudes:", list(fft_result.to_mags()))  # [1, 1, 1, 1]
```

---

### 8. Geometry

**Distance & similarity**

```python
import rmath as rm

a = rm.Vector([1, 2, 3])
b = rm.Vector([4, 5, 6])

dist = rm.geometry.euclidean_distance(a, b)
cos_sim = rm.geometry.cosine_similarity(a, b)

print("Distance:", dist)            # 5.196
print("Cosine similarity:", cos_sim) # 0.975
```

---

### 9. Interoperability

**Scikit-Learn Drop-in (Zero-Copy via `__array__`)**

```python
import rmath as rm
from sklearn.linear_model import LinearRegression

# 1. Generate data entirely in RMath (Rust)
X = rm.Array.randn(100, 1)  # 100 samples, 1 feature
Y = rm.Array.randn(100, 1)  # 100 targets

# 2. Pass RMath arrays natively into Scikit-Learn (Python)
model = LinearRegression()

# This "Just Works" because RMath natively exposes the __array__ protocol!
model.fit(X, Y)

print("Scikit-Learn R² Score:", model.score(X, Y))
```

**NumPy roundtrip**

```python
import numpy as np
import rmath as rm

np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
rm_arr = rm.Array.from_numpy(np_arr)
back = rm_arr.to_numpy()
```

**PyTorch bridge**

```python
import torch
import rmath as rm

t = torch.tensor([[1.0, 2.0]])
rm_arr = rm.Array.from_torch(t)
back = rm_arr.to_torch()
```

**pandas integration**

```python
import pandas as pd
import rmath as rm

data = rm.Array([[1, 2], [3, 4], [5, 6]])
df = data.to_dataframe(columns=["x", "y"])
back = rm.Array.from_dataframe(df)
```

---

## Performance

Benchmarked on Windows (CPython 3.13, AMD64). Medians of 20 runs, 3 warmup.

### Vector (1-D) — 167/167 tests passed

| Operation | Size | Speedup vs Python |
|-----------|------|-------------------|
| `sum_range` | 100K | **6,076x** |
| `norm_l1` | 100K | **142x** |
| `std_dev` | 100K | **119x** |
| `dot` | 100K | **43x** |
| `sin` (elementwise) | 100K | **5.8x** |
| `sort` | 100K | **4.6x** |

**Average speedup: 50x over pure Python.**

### Array (N-D) — 161/161 tests passed

| Operation | Size | Speedup vs NumPy |
|-----------|------|-------------------|
| `transpose` | 500x200 | **65x** |
| `from_numpy` | 500x200 | **38x** |
| `gelu` | 500x200 | **18x** |
| `tanh` | 500x200 | **5x** |
| `matmul` | 200x200 | competitive |

**Average speedup: 3.2x over NumPy.**

### Tensor (Autograd) — 30/30 tests passed

| Operation | Size | Speedup vs PyTorch |
|-----------|------|--------------------|
| `add` (forward) | 200x200 | **8.7x** |
| `reshape` | 200x200 | **6.6x** |
| `mul` (forward) | 200x200 | **6.2x** |
| `sigmoid` (forward) | 200x200 | **6.1x** |
| `add` (backward) | 200x200 | **5.1x** |
| `training_step` | 100x100 | **3.2x** |

**Average speedup: 3.98x over PyTorch.**

### Phase 3: Intelligence & Fusion (v0.1.5) 🚀

The latest release introduces **Single-Pass Parallel Kernels** for optimizers and elementwise math.

| Component | Operation | Gain vs Eager/PyTorch |
|-----------|-----------|-----------------------|
| **Adam** | `.step()` | **3.3x faster vs PyTorch** |
| **SGD** | `.step()` (with momentum) | **2.0x faster vs PyTorch** |
| **Linear Fusion** | `(x * 2 + 1) * 3` | **2.3x faster** |
| **Fused Reduction** | `sum(sin(x))` | **1.2x faster** |

#### Unified Lazy Engine (Loop Fusion)
RMath now supports **deferred execution** for both memory-based and disk-based arrays. Chain your operations with `.lazy()` to execute them in a single parallel pass through memory.

```python
import rmath.array as ra

# 1. In-Memory Loop Fusion (3 passes -> 1 pass)
a = ra.randn(2000, 2000)
result = a.lazy().mul(2.0).sin().exp().execute()

# 2. Disk-Streaming Fusion (Math applied during load)
result = ra.LazyArray.open("data.csv").sigmoid().add(1.0).load()
```

### Real-World Data Pipeline — rmath vs NumPy (v0.1.5)

Benchmarked on Windows (CPython 3.13, AMD64). 5 million row financial dataset.

| Pipeline Step | rmath Time | rmath Mem | NumPy Time | NumPy Mem | Speedup |
|---------------|-----------|-----------|------------|-----------|---------|
| Data Generation | 0.30s | 153 MB | 1.31s | 137 MB | **4.3× faster** |
| Data Cleaning | 0.15s | **0.5 MB** | 0.17s | 4.8 MB | **1.1× faster** |
| Feature Engineering | 0.07s | 76 MB | 0.12s | 76 MB | **1.8× faster** |
| Descriptive Stats | 0.26s | 0.07 MB | 0.25s | 0.03 MB | Comparable |
| Correlation Analysis | 0.038s | 0.04 MB | 0.43s | 0.13 MB | **11.2× faster** |
| Segmentation | 0.47s | 120 MB | 0.93s | 38 MB | **2.0× faster** |
| Linear Signal | 0.16s | 0.03 MB | 0.13s | 0.00 MB | NumPy slight edge |

> rmath wins on **5 of 7** pipeline stages. Data cleaning uses **9× less memory** than NumPy (0.5 MB vs 4.8 MB) thanks to zero-allocation `filter_where`. Full benchmark scripts in `benchmarks/pipeline/`.

### Numerical Accuracy

| Algorithm | Module | Guarantee |
|-----------|--------|-----------|
| Kahan compensated summation | Vector + Array | O(eps) error regardless of N |
| Welford's online variance | Vector + Array | Single-pass, no catastrophic cancellation |
| Parallel Kahan | Array (N >= 8K) | Chunked Kahan + merge, same accuracy as serial |

---

## How it works

```
Python ─── PyO3 FFI ──> Rust core (rayon + faer)
                            |
                 +----------+----------+
                 v          v          v
              Stack       Heap       Mmap
            (<=32 f64)  (Arc-shared) (lazy I/O)
```

- **GIL-free**: All Vector reductions, operators, norms, sorting, and statistics
  release the GIL via `py.allow_threads()`. Tensor division backward pass runs
  in pure Rust with no `Python::with_gil` re-entry.
- **Storage tiering**: Vectors with 32 or fewer elements live on the stack
  (zero allocation). Larger vectors use `Arc<Vec<f64>>` for cheap cloning.
- **Parallelism**: Rayon parallel iterators activate at 8,192 elements
  (unified threshold across Vector and Array). Below that threshold, serial
  iterators avoid thread-pool overhead.
- **Autograd**: Tensor reads data through `Arc<RwLock>` with no deep clones on
  `.shape`, `.dtype`, forward ops, or backward passes.
- **Interop**: `to_torch()` and `to_jax()` route through NumPy arrays
  (single memcpy) instead of N individual Python float allocations.
- **Type stubs**: Full `.pyi` stubs ship with the package for IDE
  autocompletion and type-checking.

## Documentation

Full API reference: **[ay-developerweb.github.io/rmath/portal/](https://ay-developerweb.github.io/rmath/portal/)**

## Author

**Ayomide Adediran** (@Ay-developerweb)
- GitHub: [Ay-developerweb](https://github.com/Ay-developerweb)
- Email: ayomideadediran45@gmail.com

## Contributing

RMath is built in Rust (`src/`) and exposed to Python via PyO3.

- **Rust source**: `src/` — core numerical engines
- **Python stubs**: `rmath/*.pyi` — type annotations
- **Benchmarks**: `benchmarks/` — automated performance suite

## License

[MIT](LICENSE)


<p align="center">
  <img src="docs/portal/rmath_logo.svg" width="180" alt="RMath Logo">
</p>

# RMath: Silicon-Speed Numerical Computing for Python

[![CI & Publish](https://github.com/Ay-developerweb/rmath/actions/workflows/publish.yml/badge.svg)](https://github.com/Ay-developerweb/rmath/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**RMath** is a high-performance numerical toolkit backed by a high-concurrency Rust engine. It provides NumPy-equivalent APIs with significant speedups in multi-threaded environments, leveraging **Rayon** for data-parallelism and **PyO3** for zero-overhead FFI.

## 🚀 Why RMath?

*   **Zero-GIL Parallelism**: Perform heavy matrix operations and statistical calculations without blocking the Python Global Interpreter Lock.
*   **Rust Precision**: Built on `faer`, `matrixmultiply`, and `ndarray` for industrial-grade accuracy.
*   **Triple-Tier Storage**: Automatic selection between Inline-Stack, Heap, and Memory-Mapped storage based on data size.
*   **No Compiler Required**: Distributed as pre-compiled wheels for all major platforms.

---

## 🛠 Features & Modules

### 1. `rmath.array` & `rmath.vector`
Core N-dimensional array processing engine.
```python
import rmath as rm
import rmath.vector as rv

# SIMD-accelerated vector ops
v1 = rv.Vector([1.0, 2.0, 3.0])
v2 = v1.exp().sin() # Parallelized element-wise ops
```

### 2. `rmath.linalg`
Advanced matrix solvers and decompositions.
*   **Solvers**: LU, QR, Cholesky, and SVD.
*   **Inversion**: High-speed matrix inversion via partial pivoting.

### 3. `rmath.stats`
Real-time descriptive and inferential statistics.
*   Uses **Welford’s Algorithm** for single-pass, numerically stable variance/mean calculation.
*   Full support for regression, p-value approximations, and distributions (Normal, Gamma, Beta).

---

## 📦 Installation

Install the pre-compiled binaries via pip:

```bash
pip install rmath
```

---

## 📖 Modern Documentation Portal

For the full API reference, architectural deep-dives, and performance benchmarks, visit our premium documentation portal:

👉 **[https://Ay-developerweb.github.io/rmath/](https://Ay-developerweb.github.io/rmath/)**

---

## ⚡ Quick Start

```python
import rmath as rm

# Generate a 1000x1000 matrix
arr = rm.Array.randn(1000, 1000)

# High-speed parallel stats
avg = arr.mean()
std = arr.std()

# Linear Algebra Solver
b = rm.Array.ones(1000, 1)
x = rm.linalg.solve(arr, b)

print(f"Residual Sum: {x.sum()}")
```

## 🤝 Contributing
RMath is built in Rust (`src/`) and exposed to Python via PyO3. 
- **Rust**: Core logic, SIMD, and Parallelism.
- **Python**: High-level API and Type Stubs (`.pyi`).

---

## 📜 License
Dual-licensed under **MIT** and **Apache 2.0**.

<p align="center">
  <img src="docs/portal/rmath_logo.svg" width="160" alt="RMath Logo">
</p>

# RMath 🚀

**High-performance numerical toolkit for Python, backed by Rust and Rayon.**

RMath is designed for developers who need NumPy-like performance with the memory safety and concurrency of Rust. It features a zero-GIL parallel execution engine, triple-tier array storage, and a premium mathematical suite.

## 📦 Installation

No Rust? No problem. RMath is distributed as pre-compiled wheels for Windows, Linux, and macOS.

```bash
pip install rmath
```

## 🌟 Key Features

- **Blazing Fast**: SIMD-parallel operations across all cores.
- **Zero-GIL**: Perform heavy math without blocking the Python interpreter.
- **Exhaustive Suite**: Vectors, N-dimensional Arrays, Stats, Calculus, Geometry, Signal Processing, and Linalg.
- **Modern Design**: Built with a focus on developer experience.

## 📖 Premium Documentation

We have a high-end, interactive documentation portal available at:
**[https://yourusername.github.io/rmath_rp/](https://yourusername.github.io/rmath_rp/)**

*(Or view the local version in `docs/portal/index.html`)*

## 🛠 Quickstart

```python
import rmath as rm

# Create a million-element array
data = rm.Array.linspace(0, 100, 1_000_000)

# Parallel compute: sin(x) * exp(x)
# Runs across all CPU cores automatically!
result = data.sin().exp().sum()

print(f"Computed Result: {result}")
```

## 📜 License
MIT / Apache-2.0

# RMath: Documentation & Hub 🚀

Welcome to the official RMath documentation. RMath is a state-of-the-art numerical engine built in Rust, optimized for Python.

## 🚀 Quick Start

```python
import rmath as rm

# Create a 2D array
a = rm.Array([[1, 2], [3, 4]])

# High-speed elementwise operations
b = a.exp().tanh()

# Linear Algebra in a single sweep
inv = b.linalg.inv()
```

## 🏗️ Core Data Structures

### 1. Array (N-Dimensional)
The multi-threaded powerhouse of RMath. 
- **Auto-Broadcasting:** Mix scalars and arrays seamlessly.
- **Copy-on-Write:** Efficient memory management for large datasets.
- **Zero-Copy Interop:** Instantly switch between RMath and NumPy.

### 2. Vector (1-Dimensional)
Specialized for 1D math. Under the hood, this is a refined subset of the Array engine, optimized for linear scan performance.

## 📈 Specialized Modules

### `rmath.linalg`
Advanced solvers: QR, SVD, Cholesky, and Eigh. Uses the `faer` backend in Rust for world-class matrix performance.

### `rmath.nn`
Neural network building blocks:
- **Activations:** GELU, ReLU, Softmax, Tanh.
- **Losses:** MSE, Cross-Entropy.
- **Layers:** BatchNorm, LayerNorm, Dropout.

### `rmath.stats`
Statistical inferential engine: Mean, Var, Std, and P-value approximations.

## 🏎️ Performance Benchmarks (vs NumPy)

| Operation | RMath | NumPy | Speedup |
|-----------|-------|-------|---------|
| GELU      | 2.1ms | 17.4ms| **8.4x**|
| SVD (200) | 42.3ms| 69.5ms| **1.6x**|
| Recip     | 125us | 1.3ms | **11x** |

## 🛠️ Installation

```bash
pip install rmath
```
*Note: Binaries are pre-compiled. No Rust installation required for end-users.*

---
© 2026 RMath Development Team. Documentation generated via RMath-DocGen.

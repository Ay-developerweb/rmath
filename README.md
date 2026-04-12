<p align="center">
  <img src="docs/portal/rmath_logo.svg" width="120" alt="RMath">
</p>

<h3 align="center">Numerical computing for Python, powered by Rust.</h3>

<p align="center">
  <a href="https://pypi.org/project/rmath-py/"><img src="https://img.shields.io/pypi/v/rmath-py?color=blue" alt="PyPI"></a>
  <a href="https://github.com/Ay-developerweb/rmath/actions/workflows/publish.yml"><img src="https://github.com/Ay-developerweb/rmath/actions/workflows/publish.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/rmath-py" alt="Python"></a>
</p>

---

RMath is a numerical toolkit that runs heavy math in Rust and exposes it to
Python via [PyO3](https://pyo3.rs). Array operations, linear algebra, stats,
calculus, and signal processing all execute **outside the GIL** on a
[Rayon](https://github.com/rayon-rs/rayon) thread pool.

```python
import rmath as rm

data = rm.Array.randn(1000, 1000)
avg, std = data.mean(), data.std()

b = rm.Array.ones(1000, 1)
x = rm.linalg.solve(data, b)
```

## Install

```bash
pip install rmath-py
```

Pre-built wheels are available for Windows, Linux, and macOS.
No Rust toolchain required.

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

## Quick examples

**Vector operations**
```python
import rmath.vector as rv

v = rv.Vector.linspace(0, 10, 1_000_000)
result = v.sin().exp().sum()   # runs on all cores
```

**Statistics**
```python
import rmath as rm

data = rm.Array.randn(10_000, 1)
report = rm.stats.describe(data)  # mean, var, skew, kurtosis
```

**Automatic differentiation**
```python
import rmath.calculus as rc

# f(x) = x² + 3x at x = 2
result = rc.Dual(2.0, 1.0)
out = result * result + result * 3.0
# out.re = 10.0, out.eps = 7.0 (derivative)
```

## How it works

```
Python (rmath)  ──PyO3 FFI──▸  Rust core (rayon + faer + ndarray)
                                  │
                      ┌───────────┼───────────┐
                      ▼           ▼           ▼
                   Stack       Heap        Mmap
                  (inline)   (shared)    (lazy I/O)
```

- **No GIL**: Heavy loops release the GIL and fan out across cores via Rayon.
- **Storage tiering**: Small arrays live on the stack, large ones on the heap,
  and huge datasets use memory-mapped files automatically.
- **Type stubs included**: Full `.pyi` stubs ship with the package for
  IDE autocompletion and type-checking.

## Documentation

Full API reference: **[ay-developerweb.github.io/rmath/portal/](https://ay-developerweb.github.io/rmath/portal/)**

## Contributing

RMath is built in Rust (`src/`) and exposed to Python via PyO3.

- **Rust source**: `src/` — core numerical engines
- **Python stubs**: `rmath/*.pyi` — type annotations
- **Docs portal**: `docs/portal/` — static HTML documentation

## License

[MIT](LICENSE)

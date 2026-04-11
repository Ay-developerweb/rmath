# Rmath Toolkit: High-Performance Numerical Engine

Rmath is a professional-grade mathematical toolkit built in Rust and exposed to Python with zero-copy efficiency. It leverages **Rayon** for multi-core parallelism, **rustfft** for spectral analysis, and **faer** for linear algebra.

## 🛠️ Installation & Build

Ensure you have the Rust toolchain installed.

```bash
# Clone and enter the directory
cd rmath

# Create a virtual environment
python -m venv venv
source venv/bin/scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install maturin polars pandas rich colorama numpy scipy

# Build the Rust core (Release mode is MANDATORY for performance)
maturin develop --release
```

## 🚀 Rmath Analyzer CLI

The project includes a modular CLI suite located in `projects/rmath_cli`.

### 1. Data Generation
Generate millions of rows of realistic sensor data in seconds.
```bash
python -m projects.rmath_cli generate --n 1000000 --out logs.csv
```

### 2. Advanced Data Analysis (Polars Integration)
Analyze massive CSVs with parallel statistics and inferential T-Tests.
```bash
python -m projects.rmath_cli analyze logs.csv --col temperature --vs pressure
```

### 3. Engineering & Signals
Perform FFTs on complex signals or solve systems of linear equations.
```bash
python -m projects.rmath_cli eng --mode fft --n 8192
python -m projects.rmath_cli eng --mode matrix
```

### 4. Spatial Geometry
Compute convex hulls and perform spatial containment checks.
```bash
python -m projects.rmath_cli geo --n 5000
```

## 📊 Benchmarking

Validate performance parity with NumPy and SciPy using the automated suites:

```bash
python -m benchmarks.benchmark_vector    # Vectorized arithmetic (avg 20x speedup)
python -m benchmarks.benchmark_stats     # Parallel statistics (avg 120x speedup)
python -m benchmarks.benchmark_signal    # FFT & Convolution (O(N log N) speedup)
python -m benchmarks.benchmark_geometry  # Spatial topology & Kernels
```

## 🏛️ Project Architecture

- `src/vector/`: Parallelized 1D vector operations (SIMD-ready).
- `src/stats/`: Descriptive & Inferential engine (Welford & Welch).
- `src/signal/`: Iterative FFT and spectral convolution.
- `src/geometry/`: Spatial kernels with serial fallbacks for small vectors.
- `src/calculus/`: Automatic Differentiation (Dual Numbers) and Integration.

---
**Performance Note**: Always use the `.lazy()` bridge for complex chains to benefit from fused kernel execution.

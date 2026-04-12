"""
RMath — High-Performance Numerical Toolkit for Python (Rust-backed).

A professional-grade suite for vectorized math, statistical analysis, 
linear algebra, and automatic differentiation. Powered by a multi-tiered 
Rust engine and parallelized with Rayon for zero-GIL performance.

Modules:
    - array: N-dimensional parallel arrays (Tensor-like).
    - vector: High-speed 1D signal processing and reducers.
    - linalg: BLAS-accelerated linear algebra solvers.
    - stats: Descriptive and inferential statistical engines.
    - calculus: Automatic differentiation (AD) and integration.
"""
from . import scalar as scalar
from . import stats as stats
from . import vector as vector
from . import array as array
from . import constants as constants
from . import linalg as linalg
from . import calculus as calculus

# Explicitly re-export classes for root access
from .vector import Vector as Vector
from .array import Array as Array

__all__ = [
    "scalar", "stats", "vector", "array", "constants", "linalg", "calculus",
    "Vector", "Array"
]

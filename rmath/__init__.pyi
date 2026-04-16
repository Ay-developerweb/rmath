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
from . import nn as nn
from . import geometry as geometry
from . import special as special
from . import signal as signal

# Explicitly re-export classes for root access
from .vector import Vector as Vector
from .array import Array as Array, Tensor as Tensor, LazyArray as LazyArray
from .scalar import Scalar as Scalar
from .calculus import Dual as Dual

from typing import Union, Sequence, List
def sum(v: Union[Vector, Sequence[float]]) -> float: ...
def mean(v: Union[Vector, Sequence[float]]) -> float: ...
def min(v: Union[Vector, Sequence[float]]) -> float: ...
def max(v: Union[Vector, Sequence[float]]) -> float: ...
def loop_range(start: int, end: int) -> 'scalar.LazyPipeline': ...

__all__ = [
    "scalar", "stats", "vector", "array", "constants", "linalg", "calculus",
    "nn", "geometry", "special", "signal",
    "Vector", "Array", "Tensor", "LazyArray", "Scalar", "Dual",
    "sum", "mean", "min", "max", "loop_range"
]

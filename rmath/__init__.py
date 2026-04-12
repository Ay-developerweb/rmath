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
from ._rmath import (
    loop_range
)
# Import our new package-modules (for hovers/docs)
from .vector import Vector
from .array import Array
from .scalar import Scalar
from .calculus import Dual

from . import vector
from . import array
from . import stats
from . import linalg
from . import calculus
from . import scalar
from . import nn
from . import geometry
from . import special
from . import signal
from . import constants

# Standard __all__ for clean exports
__all__ = [
    "Array", "Vector", "Scalar",
    "scalar", "stats", "vector", "array", "constants",
    "geometry", "linalg", "nn", "special", "signal",
    "calculus", "loop_range"
]

# Cleanup: Remove modules that might have been imported during initialization
# to keep the root namespace pristine.
import sys as _sys
_this_mod = _sys.modules[__name__]
if hasattr(_this_mod, "rmath"):
    delattr(_this_mod, "rmath")
if hasattr(_this_mod, "_sys"):
    delattr(_this_mod, "_sys")
if hasattr(_this_mod, "_this_mod"):
    delattr(_this_mod, "_this_mod")

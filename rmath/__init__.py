# RMath: High-Performance Mathematical Toolkit
from ._rmath import (
    Array, Vector, Scalar, 
    scalar, stats, vector, array, 
    constants, geometry, linalg, nn,
    special, signal, calculus, loop_range
)

# Standard __all__ for clean exports
__all__ = [
    "Array", "Vector", "Scalar",
    "scalar", "stats", "vector", "array",
    "constants", "geometry", "linalg", "nn",
    "special", "signal", "calculus", "loop_range"
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

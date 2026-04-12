"""
rmath.scalar — high-performance scalar math operations backed by Rust.

Mirrors Python's ``math`` module for scalar float operations, plus extras:
clamp, lerp, fma, and integer bit operations.
"""
from .._rmath import scalar as _scalar

# Re-export everything from the binary submodule into this package namespace
for name in dir(_scalar):
    if not name.startswith('_'):
        globals()[name] = getattr(_scalar, name)

from .._rmath import Scalar

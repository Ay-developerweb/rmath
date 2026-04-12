"""
rmath.vector — High-speed 1D signal processing and reducers.

Powered by a parallel Rust engine with Rayon. Features specialized 
storage tiers for zero-overhead math on small datasets and 
multi-core scaling for large-scale analysis.
"""
from .._rmath import vector as _vector

# Re-export everything from the binary submodule into this package namespace
for name in dir(_vector):
    if not name.startswith('_'):
        globals()[name] = getattr(_vector, name)
        
from .._rmath import Vector

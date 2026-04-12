"""
rmath.array — Parallel N-D Array engine (Tensor-like).

Row-major `Vec<f64>` backed with Rayon-parallelized element-wise ops.
Supports broadcasting, matrix multiplication, and tiered storage 
(Inline, Heap, Mmap) for scaling from L1-cache to GB-scale files.
"""
from .._rmath import array as _array

# Re-export everything from the binary submodule into this package namespace
for name in dir(_array):
    if not name.startswith('_'):
        globals()[name] = getattr(_array, name)

from .._rmath import Array

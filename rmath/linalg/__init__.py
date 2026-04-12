"""
rmath.linalg — BLAS-Accelerated Linear Algebra.

Provides highly optimized matrix operations including:
    - Matrix Inversion and Determinants.
    - Solving Linear Systems (Ax = B).
    - Decompositions (QR, SVD, Cholesky, Eigh).
    - Specialized Norm and Rank kernels.
"""
from .._rmath import linalg as _linalg

for name in dir(_linalg):
    if not name.startswith('_'):
        globals()[name] = getattr(_linalg, name)

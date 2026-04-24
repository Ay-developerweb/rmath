"""
rmath.linalg — BLAS-Accelerated Linear Algebra.

Provides highly optimized matrix operations including:
    - Inversion and Determinants.
    - Solving Linear Systems (Ax = B).
    - Decompositions (QR, SVD, Cholesky, Eigh).
    - Norms and Rank calculations.
"""
from typing import Union, Tuple, List
from .array import Array
from .vector import Vector

def inv(matrix: Array) -> Array:
    """Compute the inverse of a square matrix."""
    ...

def det(matrix: Array) -> float:
    """Compute the determinant of a square matrix."""
    ...

def solve(a: Array, b: Array) -> Array:
    """Solve the linear system Ax = B for x."""
    ...

def qr(matrix: Array) -> Tuple[Array, Array]:
    """Perform QR decomposition."""
    ...

def svd(matrix: Array) -> Tuple[Array, Vector, Array]:
    """Perform Singular Value Decomposition."""
    ...

def eigh(matrix: Array) -> Tuple[Vector, Array]:
    """Eigenvalues and eigenvectors of a symmetric matrix.
    
    Returns:
        tuple (vals, vecs)
    """
    ...

def cholesky(matrix: Array) -> Array:
    """Cholesky decomposition of a PD matrix."""
    ...

def transpose(matrix: Array) -> Array:
    """Return the transpose of a matrix."""
    ...
def rank(matrix: Array) -> int:
    """Compute the numerical rank of a matrix."""
    ...
def pseudo_inv(matrix: Array) -> Array:
    """Compute the Moore-Penrose pseudo-inverse."""
    ...
def gram_matrix(matrix: Array) -> Array:
    """Compute the Gram matrix: ``A^T @ A``."""
    ...
def covariance(matrix: Array) -> Array:
    """Compute the covariance matrix (rows = variables, columns = observations)."""
    ...

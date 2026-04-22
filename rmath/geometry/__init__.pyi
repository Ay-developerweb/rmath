"""
rmath.geometry — Distance metrics and spatial calculations.
Highly parallelized for long-vector similarity searches.
"""

from typing import List, Tuple, Sequence, Union, Optional
from rmath.vector import Vector
from rmath.array import Array

class Quaternion:
    """A Quaternion for 3D rotations: q = w + xi + yj + zk.
    
    Examples:
        >>> from rmath.geometry import Quaternion
        >>> q = Quaternion(1, 0, 0, 0)
        >>> q.rotate_vector([1, 0, 0])
        Vector([1.0, 0.0, 0.0])
    """
    w: float
    x: float
    y: float
    z: float

    def __init__(self, w: float, x: float, y: float, z: float) -> None: ...
    def norm(self) -> float: ...
    def normalize(self) -> 'Quaternion': ...
    def rotate_vector(self, v: Union[Vector, Sequence[float]]) -> Vector: ...
    def __mul__(self, other: 'Quaternion') -> 'Quaternion': ...
    def __repr__(self) -> str: ...

def euclidean_distance(v1: Union[Vector, Sequence[float]], v2: Union[Vector, Sequence[float]]) -> float:
    """Calculate the Euclidean distance (L2 norm) between two vectors."""
    ...

def manhattan_distance(v1: Union[Vector, Sequence[float]], v2: Union[Vector, Sequence[float]]) -> float:
    """Calculate the Manhattan distance (L1 norm) between two vectors."""
    ...

def minkowski_distance(v1: Union[Vector, Sequence[float]], v2: Union[Vector, Sequence[float]], p: float) -> float:
    """Calculate the Minkowski distance (Lp norm) between two vectors."""
    ...

def cosine_similarity(v1: Union[Vector, Sequence[float]], v2: Union[Vector, Sequence[float]]) -> float:
    """Calculate the cosine similarity between two vectors."""
    ...

def projection(v: Union[Vector, Sequence[float]], target: Union[Vector, Sequence[float]]) -> Vector:
    """Project vector `v` onto target vector `target`."""
    ...

def cross_product(v1: Union[Vector, Sequence[float]], v2: Union[Vector, Sequence[float]]) -> Vector:
    """Calculate the 3D cross product of two vectors."""
    ...

def angle_between(v1: Union[Vector, Sequence[float]], v2: Union[Vector, Sequence[float]]) -> float:
    """Calculate the angle in radians between two vectors."""
    ...

def cdist(a: Array, b: Array) -> Array:
    """Compute the pairwise Euclidean distance matrix between two sets of points.
    
    Args:
        a: M x D Array.
        b: N x D Array.
    """
    ...

def is_point_in_polygon(x: float, y: float, poly_x: Sequence[float], poly_y: Sequence[float]) -> bool:
    """Determine if a point (x, y) is inside a polygon."""
    ...

def convex_hull(
    arg1: Union[Array, Sequence[float]], 
    arg2: Optional[Sequence[float]] = None
) -> Union[Array, Tuple[List[float], List[float]]]:
    """Compute the Convex Hull of a set of 2D points.
    
    If arg2 is provided, arg1 and arg2 are treated as X and Y coordinate lists.
    If arg2 is None, arg1 is treated as an (N, 2) Array.
    """
    ...

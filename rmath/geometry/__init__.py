"""
rmath.geometry — High-performance Spatial Geometry.

Provides optimized calculations for:
    - Distance metrics (Euclidean, Haversine).
    - Convex Hull and Spatial intersections.
    - Mesh processing and 3D transformations.
"""
from .._rmath import geometry as _geometry

for name in dir(_geometry):
    if not name.startswith('_'):
        globals()[name] = getattr(_geometry, name)

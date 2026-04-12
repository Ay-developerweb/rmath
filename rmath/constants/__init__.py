"""
rmath.constants — Universal Mathematical and Physical Constants.

High-precision (f64) constants for scientific computing:
    - Mathematical: pi, e, tau, zeta(3).
    - Physical: G, c, h, N_A, k_B.
"""
from .._rmath import constants as _constants

for name in dir(_constants):
    if not name.startswith('_'):
        globals()[name] = getattr(_constants, name)

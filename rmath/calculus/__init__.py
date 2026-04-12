"""
rmath.calculus — Automatic Differentiation and Numerical Integration.

High-performance numerical analysis engine. Supported features:
    - Forward-mode Automatic Differentiation (AD).
    - Parallelized Integration (Simpson, Trapezoidal).
    - Root-finding (Newton, Bisection) with exact derivatives.
"""
from .._rmath import calculus as _calculus

for name in dir(_calculus):
    if not name.startswith('_'):
        globals()[name] = getattr(_calculus, name)

from .._rmath import Dual

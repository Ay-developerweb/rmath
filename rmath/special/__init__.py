"""
rmath.special — Special Mathematical Functions.

Rust-backed high-precision implementations of:
    - Gamma, Beta, and Error functions.
    - Bessel, Kelvin, and Airy functions.
    - Statistical distribution functions (PDF/CDF).
"""
from .._rmath import special as _special

for name in dir(_special):
    if not name.startswith('_'):
        globals()[name] = getattr(_special, name)

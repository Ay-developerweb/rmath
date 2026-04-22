"""
rmath.special — Special Mathematical Functions (Gamma, Erf, etc.)
Highly optimized and parallelized across all numeric containers.
"""

from typing import Union, Sequence
from rmath.vector import Vector
from rmath.array import Array

def gamma(data: Union[float, Sequence[float], Vector, Array]) -> Union[float, Vector, Array]:
    """Compute the Gamma function Γ(x).
    
    Examples:
        >>> from rmath import special
        >>> special.gamma(5.0)
        24.0
    """
    ...

def ln_gamma(data: Union[float, Sequence[float], Vector, Array]) -> Union[float, Vector, Array]:
    """Compute the natural logarithm of the Gamma function, ln|Γ(x)|."""
    ...

def erf(data: Union[float, Sequence[float], Vector, Array]) -> Union[float, Vector, Array]:
    """Compute the Error function erf(x)."""
    ...

from typing import Union, Sequence
from rmath import Vector
from rmath.array import Array

def gamma(data: Union[float, Sequence[float], Vector, Array]) -> Union[float, Vector, Array]:
    """Compute the Gamma function Γ(x)."""
    ...

def ln_gamma(data: Union[float, Sequence[float], Vector, Array]) -> Union[float, Vector, Array]:
    """Compute the natural logarithm of the Gamma function, ln|Γ(x)|."""
    ...

def erf(data: Union[float, Sequence[float], Vector, Array]) -> Union[float, Vector, Array]:
    """Compute the Error function erf(x)."""
    ...

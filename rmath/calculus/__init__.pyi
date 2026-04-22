"""
rmath.calculus — Automatic Differentiation and Numerical Integration.

Features:
    - Dual: Forward-mode AD using dual numbers (a + bε).
    - Integration: Simpson's and Trapezoidal parallelized rules.
    - Solvers: Newton-Raphson root-finding for exact derivatives.
"""
from typing import Callable, Union, Tuple, Any
from rmath.vector import Vector

class Dual:
    """
    Dual numbers: a + b*epsilon where epsilon^2 = 0.
    Used for Automatic Differentiation (AD).
    
    Examples:
        >>> from rmath.calculus import Dual
        >>> d = Dual(3.0, 1.0)
        >>> f = lambda x: x**2 + 5*x
        >>> res = f(d)
        >>> res.value
        24.0
        >>> res.derivative
        11.0
    """
    val: float
    der: float

    def __init__(self, val: float, der: float) -> None: ...
    @property
    def value(self) -> float: ...
    @property
    def derivative(self) -> float: ...

    def __repr__(self) -> str: ...
    def __add__(self, other: Union['Dual', float]) -> 'Dual': ...
    def __sub__(self, other: Union['Dual', float]) -> 'Dual': ...
    def __mul__(self, other: Union['Dual', float]) -> 'Dual': ...
    def __truediv__(self, other: Union['Dual', float]) -> 'Dual': ...
    def __pow__(self, p: float) -> 'Dual': ...
    def __neg__(self) -> 'Dual': ...

    def sin(self) -> 'Dual': ...
    def cos(self) -> 'Dual': ...
    def exp(self) -> 'Dual': ...
    def log(self) -> 'Dual': ...
    def erf(self) -> 'Dual': ...
    def gamma(self) -> 'Dual': ...

def integrate_trapezoidal(x: Vector, y: Vector) -> float:
    """Compute definite integral using the trapezoidal rule.
    
    Examples:
        >>> from rmath.vector import Vector
        >>> from rmath.calculus import integrate_trapezoidal
        >>> x = Vector([0, 1, 2])
        >>> y = Vector([1, 1, 1])
        >>> integrate_trapezoidal(x, y)
        2.0
    """
    ...

def integrate_simpson_array(y: Vector, dx: float) -> float:
    """Compute definite integral of evenly spaced data using Simpson's Rule."""
    ...

def integrate_simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Compute definite integral of a function f from a to b using Simpson's Rule.
    
    Examples:
        >>> from rmath.calculus import integrate_simpson
        >>> integrate_simpson(lambda x: x**2, 0, 1, 100)
        0.3333333333333333
    """
    ...

def find_root_newton(f: Callable[[Dual], Dual], x0: float, tol: float = 1e-7, max_iter: int = 50) -> float:
    """Find root of f(x) = 0 using Newton's method and AD.
    
    Examples:
        >>> from rmath.calculus import find_root_newton
        >>> find_root_newton(lambda x: x**2 - 2, 1.5)
        1.4142135623746899
    """
    ...

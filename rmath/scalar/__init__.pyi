"""
rmath.scalar — high-performance scalar math operations backed by Rust.

Mirrors Python's ``math`` module for scalar float operations, plus extras:
clamp, lerp, fma, and integer bit operations.

Import style::

    import rmath.scalar as rs
    rs.add(1.0, 2.0)  # 3.0

All functions accept and return ``float`` unless noted otherwise.
Arithmetic errors raise specific Python exceptions:

* ``ZeroDivisionError`` — division or modulo by zero (div)
* ``ValueError``        — domain errors (sqrt of negative, log of zero, etc.)
* ``OverflowError``     — result exceeds float range

Note:
    ``remainder(x, y)`` follows Python's ``%`` semantics (sign follows divisor).
    ``fmod(x, y)``      follows C/Java semantics   (sign follows dividend).
"""

from typing import Optional, Union, Tuple, List, Sequence, Any

# ── Classes ──────────────────────────────────────────────────────────────────

class Scalar:
    """A high-performance, Rust-backed scalar numeric value.

    Wraps a 64-bit IEEE-754 float and exposes it to Python with a complete
    numeric protocol: arithmetic, comparison, hashing, and boolean coercion.
    """

    def __init__(self, value: float) -> None:
        """Create a new high-precision Scalar from a float."""
        ...

    def to_python(self) -> float:
        """Convert to a native Python float."""
        ...

    # --- Protocol ---
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __format__(self, format_spec: str) -> str: ...
    def __hash__(self) -> int: ...

    # --- Comparison ---
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __lt__(self, other: Union[float, "Scalar"]) -> bool: ...
    def __le__(self, other: Union[float, "Scalar"]) -> bool: ...
    def __gt__(self, other: Union[float, "Scalar"]) -> bool: ...
    def __ge__(self, other: Union[float, "Scalar"]) -> bool: ...

    # --- Arithmetic ---
    def __add__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __radd__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __sub__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __rsub__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __mul__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __rmul__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __truediv__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __rtruediv__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __floordiv__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __rfloordiv__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __mod__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __rmod__(self, other: Union[float, "Scalar"]) -> "Scalar": ...
    def __pow__(self, other: Union[float, "Scalar"], modulo: Optional[Any] = None) -> "Scalar": ...
    def __rpow__(self, other: Union[float, "Scalar"], modulo: Optional[Any] = None) -> "Scalar": ...
    def __neg__(self) -> "Scalar": ...
    def __pos__(self) -> "Scalar": ...
    def __abs__(self) -> "Scalar": ...

    # --- Math Methods ---
    def sqrt(self) -> "Scalar":
        """Square root. Raises ValueError for negative input."""
        ...
    def cbrt(self) -> "Scalar":
        """Cube root. Defined for all real numbers."""
        ...
    def pow(self, exp: float) -> "Scalar":
        """Raise to a real power. Raises ValueError if the result is NaN."""
        ...
    def exp(self) -> "Scalar":
        """Natural exponential: e^x."""
        ...
    def exp2(self) -> "Scalar":
        """Base-2 exponential: 2^x."""
        ...
    def log(self, base: Optional[float] = None) -> "Scalar":
        """Natural logarithm (base e), or arbitrary base if provided."""
        ...
    def log2(self) -> "Scalar":
        """Base-2 logarithm."""
        ...
    def log10(self) -> "Scalar":
        """Base-10 logarithm."""
        ...
    def sin(self) -> "Scalar":
        """Sine (input in radians)."""
        ...
    def cos(self) -> "Scalar":
        """Cosine (input in radians)."""
        ...
    def tan(self) -> "Scalar":
        """Tangent (input in radians)."""
        ...
    def asin(self) -> "Scalar":
        """Arcsine (returns radians)."""
        ...
    def acos(self) -> "Scalar":
        """Arccosine (returns radians)."""
        ...
    def atan(self) -> "Scalar":
        """Arctangent."""
        ...
    def atan2(self, x: float) -> "Scalar":
        """Two-argument arctangent."""
        ...
    def sinh(self) -> "Scalar":
        """Hyperbolic sine."""
        ...
    def cosh(self) -> "Scalar":
        """Hyperbolic cosine."""
        ...
    def tanh(self) -> "Scalar":
        """Hyperbolic tangent."""
        ...
    def asinh(self) -> "Scalar":
        """Inverse hyperbolic sine."""
        ...
    def acosh(self) -> "Scalar":
        """Inverse hyperbolic cosine."""
        ...
    def atanh(self) -> "Scalar":
        """Inverse hyperbolic tangent."""
        ...
    def ceil(self) -> "Scalar":
        """Round toward positive infinity."""
        ...
    def floor(self) -> "Scalar":
        """Round toward negative infinity."""
        ...
    def round(self) -> "Scalar":
        """Round to the nearest integer, ties away from zero."""
        ...
    def trunc(self) -> "Scalar":
        """Truncate toward zero."""
        ...
    def fract(self) -> "Scalar":
        """Fractional part: self - self.trunc()."""
        ...
    def abs(self) -> "Scalar":
        """Absolute value."""
        ...
    def hypot(self, other: float) -> "Scalar":
        """Numerically stable sqrt(self² + other²)."""
        ...

    # --- Predicates ---
    def is_nan(self) -> bool:
        """Returns True if this value is NaN."""
        ...
    def is_inf(self) -> bool:
        """Returns True if this value is infinite."""
        ...
    def is_finite(self) -> bool:
        """Returns True if this value is neither NaN nor infinite."""
        ...

    # --- Utility ---
    def clamp(self, low: float, high: float) -> "Scalar":
        """Clamps the value to [low, high]."""
        ...
    def signum(self) -> "Scalar":
        """Returns the sign (-1, 0, or 1)."""
        ...
    def lerp(self, other: float, t: float) -> "Scalar":
        """Linear interpolation: self + t * (other - self)."""
        ...


class Complex:
    """A high-performance complex number (re + im*j) backed by Rust."""

    def __init__(self, re: float, im: float) -> None:
        """Create a new complex number."""
        ...

    @staticmethod
    def from_polar(r: float, theta: float) -> "Complex":
        """Construct from polar coordinates (modulus, argument)."""
        ...

    @property
    def re(self) -> float: ...
    @re.setter
    def re(self, value: float) -> None: ...
    @property
    def im(self) -> float: ...
    @im.setter
    def im(self, value: float) -> None: ...

    def abs(self) -> float:
        """Modulus |z|."""
        ...
    def arg(self) -> float:
        """Argument (phase angle) in radians."""
        ...
    def conjugate(self) -> "Complex":
        """Complex conjugate."""
        ...
    def to_polar(self) -> Tuple[float, float]:
        """Convert to polar form: (modulus, argument)."""
        ...

    # --- Math ---
    def exp(self) -> "Complex": ...
    def log(self) -> "Complex": ...
    def sqrt(self) -> "Complex": ...
    def pow(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def sin(self) -> "Complex": ...
    def cos(self) -> "Complex": ...

    # --- Protocol ---
    def __abs__(self) -> float: ...
    def __neg__(self) -> "Complex": ...
    def __pos__(self) -> "Complex": ...
    def __bool__(self) -> bool: ...
    def __complex__(self) -> Tuple[float, float]: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

    # --- Comparison (by modulus) ---
    def __lt__(self, other: Union[float, "Complex", Scalar]) -> bool: ...
    def __le__(self, other: Union[float, "Complex", Scalar]) -> bool: ...
    def __gt__(self, other: Union[float, "Complex", Scalar]) -> bool: ...
    def __ge__(self, other: Union[float, "Complex", Scalar]) -> bool: ...

    # --- Arithmetic ---
    def __add__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __radd__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __sub__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __rsub__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __mul__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __rmul__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __truediv__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __rtruediv__(self, other: Union[float, "Complex", Scalar]) -> "Complex": ...
    def __pow__(self, other: Union[float, "Complex", Scalar], modulo: Optional[Any] = None) -> "Complex": ...
    def __rpow__(self, other: Union[float, "Complex", Scalar], modulo: Optional[Any] = None) -> "Complex": ...


class LazyPipeline:
    """A lazy, parallel evaluation pipeline for scalar operations."""

    def to_vector(self) -> Any:
        """Materialise the pipeline into a Vector."""
        ...
    def to_tuple(self) -> Tuple[float, ...]:
        """Materialise the pipeline into a Python tuple of floats."""
        ...
    def sum(self) -> float:
        """Sum all elements."""
        ...
    def mean(self) -> float:
        """Arithmetic mean."""
        ...
    def var(self) -> float:
        """Population variance."""
        ...
    def std(self) -> float:
        """Standard deviation."""
        ...
    def max(self) -> float:
        """Maximum value."""
        ...
    def min(self) -> float:
        """Minimum value."""
        ...

    # --- Chaining (Mapping) ---
    def sin(self) -> "LazyPipeline": ...
    def cos(self) -> "LazyPipeline": ...
    def sqrt(self) -> "LazyPipeline": ...
    def abs(self) -> "LazyPipeline": ...
    def exp(self) -> "LazyPipeline": ...
    def add(self, val: float) -> "LazyPipeline": ...
    def mul(self, val: float) -> "LazyPipeline": ...
    def sub(self, val: float) -> "LazyPipeline": ...
    def div(self, val: float) -> "LazyPipeline": ...

    # --- Chaining (Filtering) ---
    def filter_gt(self, val: float) -> "LazyPipeline": ...
    def filter_lt(self, val: float) -> "LazyPipeline": ...
    def filter_finite(self) -> "LazyPipeline": ...

    def as_(self, target: str) -> "LazyPipeline": ...

    def __len__(self) -> int:
        """Returns the source length before filters."""
        ...
    def __repr__(self) -> str: ...


# ── Functional API ───────────────────────────────────────────────────────────

# ── Arithmetic ────────────────────────────────────────────────────────────────

def add(x: float, y: float) -> float:
    """Return ``x + y``.

    Examples:
        >>> rs.add(10.5, 0.5)
        11.0
        >>> rs.add(float('inf'), 1.0)
        inf
    """
    ...

def sub(x: float, y: float) -> float:
    """Return ``x - y``.

    Examples:
        >>> rs.sub(1.0, 0.1)
        0.9
    """
    ...

def mul(x: float, y: float) -> float:
    """Return ``x * y``.

    Examples:
        >>> rs.mul(2.5, 4.0)
        10.0
    """
    ...

def div(x: float, y: float) -> float:
    """Return ``x / y``.

    Raises:
        ZeroDivisionError: If ``y`` is ``0.0``.

    Examples:
        >>> rs.div(1.0, 4.0)
        0.25
        >>> rs.div(1.0, 0.0)
        ZeroDivisionError: division by zero
    """
    ...

def fmod(x: float, y: float) -> float:
    """Return the C-style remainder of ``x / y``.

    The sign of the result follows the **dividend** (``x``), matching
    ``math.fmod``, C, and Java. Compare with ``remainder``, where the
    sign follows the divisor.

    Examples:
        >>> rs.fmod(-5.0, 3.0)
        -2.0
        >>> rs.fmod(5.0, 3.0)
        2.0
    """
    ...

def remainder(x: float, y: float) -> float:
    """Return the floor-division remainder of ``x / y``.

    The sign of the result follows the **divisor** (``y``), matching
    Python's built-in ``%`` operator. Compare with ``fmod``, where the
    sign follows the dividend.

    Examples:
        >>> rs.remainder(-5.0, 3.0)
        1.0
        >>> rs.remainder(5.0, 3.0)
        2.0
    """
    ...

# ── Rounding & range ──────────────────────────────────────────────────────────

def ceil(x: float) -> float:
    """Return the smallest integer >= x as a float.

    Examples:
        >>> rs.ceil(2.0001)
        3.0
        >>> rs.ceil(-1.5)
        -1.0
    """
    ...

def floor(x: float) -> float:
    """Return the largest integer <= x as a float.

    Examples:
        >>> rs.floor(2.999)
        2.0
        >>> rs.floor(-1.5)
        -2.0
    """
    ...

def trunc(x: float) -> float:
    """Return x truncated toward zero (fractional part removed).

    Unlike floor, always moves toward zero regardless of sign.

    Examples:
        >>> rs.trunc(2.9)
        2.0
        >>> rs.trunc(-2.9)
        -2.0
    """
    ...

def round(x: float) -> float:
    """Return x rounded to the nearest integer, half away from zero.

    Note:
        This differs from Python's built-in round(), which uses
        half-to-even (banker's rounding). Use round_half_even()
        for that behaviour.

    Examples:
        >>> rs.round(2.5)
        3.0
        >>> rs.round(-2.5)
        -3.0
    """
    ...

def round_half_even(x: float) -> float:
    """Return x rounded to the nearest integer, half to even (banker's rounding).

    When x is exactly halfway between two integers, rounds to the
    nearest even integer to reduce cumulative statistical bias.

    Examples:
        >>> rs.round_half_even(2.5)
        2.0
        >>> rs.round_half_even(3.5)
        4.0
        >>> rs.round_half_even(-2.5)
        -2.0
    """
    ...

def signum(x: float) -> float:
    """Return the sign of x as -1.0, 0.0, or 1.0.

    Examples:
        >>> rs.signum(-42.0)
        -1.0
        >>> rs.signum(0.0)
        0.0
        >>> rs.signum(99.0)
        1.0
    """
    ...

def abs(x: float) -> float:
    """Return the absolute value of x.

    Examples:
        >>> rs.abs(-5.0)
        5.0
        >>> rs.abs(0.0)
        0.0
    """
    ...

def clamp(x: float, min: float, max: float) -> float:
    """Return x clamped to the range [min, max].

    Raises:
        ValueError: If min > max.

    Examples:
        >>> rs.clamp(10.0, 0.0, 5.0)
        5.0
        >>> rs.clamp(-10.0, 0.0, 5.0)
        0.0
        >>> rs.clamp(3.0, 0.0, 5.0)
        3.0
    """
    ...

def lerp(a: float, b: float, t: float) -> float:
    """Return the linear interpolation between a and b at parameter t.

    Computes a + t * (b - a). No clamping is applied — values of t
    outside [0.0, 1.0] extrapolate beyond the [a, b] range.

    Examples:
        >>> rs.lerp(10.0, 20.0, 0.5)
        15.0
        >>> rs.lerp(0.0, 100.0, 1.5)
        150.0
    """
    ...

# ── Roots, powers & abs ───────────────────────────────────────────────────────

def sqrt(x: float) -> float:
    """Return the non-negative square root of x.

    Raises:
        ValueError: If x is negative.
    """
    ...

def cbrt(x: float) -> float:
    """Return the real cube root of x. Defined for all real numbers."""
    ...

def root(x: float, n: int) -> float:
    """Return the real principal n-th root of x.

    Handles odd roots of negative numbers correctly.
    Negative n computes the reciprocal root (e.g. n=-3 gives 1/cbrt(x)).

    Raises:
        ValueError: If n is 0, or n is even and x is negative.
    """
    ...

def pow(x: float, y: float) -> float:
    """Return x raised to the power y.

    Raises:
        ValueError: If the result would be NaN (e.g. pow(-2.0, 0.5)).
    """
    ...

def inv_sqrt(x: float) -> float:
    """Return 1.0 / sqrt(x). Optimised for vector normalisation.

    Raises:
        ValueError: If x is zero or negative.
    """
    ...      # 1/sqrt(x); raises ValueError if x <= 0
def hypot(x: float, y: float) -> float: ...
def hypot_3d(x: float, y: float, z: float) -> float: ...
def fma(x: float, y: float, z: float) -> float: ...  # x*y + z  (fused multiply-add)

# ── Exponential & logarithm ───────────────────────────────────────────────────

def exp(x: float) -> float:
    """Return e ** x."""
    ...

def exp2(x: float) -> float:
    """Return 2.0 ** x."""
    ...

def expm1(x: float) -> float:
    """Return exp(x) - 1, accurate for x near zero."""
    ...

def log(x: float, base: float | None = None) -> float:
    """Return the logarithm of x to the given base, or ln(x) if base is omitted.

    Raises:
        ValueError: If x <= 0.0, or base <= 0.0, or base == 1.0.
    """
    ...

def log2(x: float) -> float:
    """Return the base-2 logarithm of x.

    Raises:
        ValueError: If x <= 0.0.
    """
    ...

def log10(x: float) -> float:
    """Return the base-10 logarithm of x.

    Raises:
        ValueError: If x <= 0.0.
    """
    ...

def log1p(x: float) -> float:
    """Return ln(1 + x), accurate for x near zero.

    Raises:
        ValueError: If x <= -1.0.
    """
    ...

def logsumexp2(x: float, y: float) -> float:
    """Return ln(exp(x) + exp(y)), computed in a numerically stable way."""
    ...

# ── Trigonometry ──────────────────────────────────────────────────────────────
# ── Trigonometry ──────────────────────────────────────────────────────────────

def sin(x: float) -> float:
    """Return sin(x) for x in radians."""
    ...

def cos(x: float) -> float:
    """Return cos(x) for x in radians."""
    ...

def tan(x: float) -> float:
    """Return tan(x) for x in radians.

    Note:
        Due to float precision, tan(pi/4) returns 0.9999999999999999
        rather than exactly 1.0.
    """
    ...

def asin(x: float) -> float:
    """Return asin(x) in radians, result in [-pi/2, pi/2].

    Raises:
        ValueError: If x is outside [-1.0, 1.0].
    """
    ...

def acos(x: float) -> float:
    """Return acos(x) in radians, result in [0, pi].

    Raises:
        ValueError: If x is outside [-1.0, 1.0].
    """
    ...

def atan(x: float) -> float:
    """Return atan(x) in radians, result in (-pi/2, pi/2)."""
    ...

def atan2(y: float, x: float) -> float:
    """Return atan2(y, x) in radians, result in (-pi, pi].

    Correctly handles x=0 and determines quadrant from signs of both
    arguments. atan2(0.0, 0.0) returns 0.0 by convention.
    """
    ...

# ── Inverse hyperbolic ────────────────────────────────────────────────────────

def asinh(x: float) -> float:
    """Return the inverse hyperbolic sine of x. Defined for all reals."""
    ...

def acosh(x: float) -> float:
    """Return the inverse hyperbolic cosine of x.

    Raises:
        ValueError: If x < 1.0.
    """
    ...

def atanh(x: float) -> float:
    """Return the inverse hyperbolic tangent of x.

    Raises:
        ValueError: If x <= -1.0 or x >= 1.0.
    """
    ...
def hypot(x: float, y: float) -> float:
    """Return sqrt(x² + y²), stable against intermediate overflow."""
    ...

def sinh(x: float) -> float:
    """Return the hyperbolic sine of x."""
    ...

def cosh(x: float) -> float:
    """Return the hyperbolic cosine of x."""
    ...

def tanh(x: float) -> float:
    """Return the hyperbolic tangent of x, result always in (-1, 1)."""
    ...
def sin(x: float) -> float: ...
def cos(x: float) -> float: ...
def tan(x: float) -> float: ...

def asin(x: float) -> float: ...
def acos(x: float) -> float: ...
def atan(x: float) -> float: ...
def atan2(y: float, x: float) -> float: ...
def degrees(x: float) -> float: ...
def radians(x: float) -> float: ...

# ── Hyperbolic ────────────────────────────────────────────────────────────────
# ── Inverse hyperbolic ────────────────────────────────────────────────────────

def asinh(x: float) -> float:
    """Return the inverse hyperbolic sine of x. Defined for all reals."""
    ...

def acosh(x: float) -> float:
    """Return the inverse hyperbolic cosine of x.

    Raises:
        ValueError: If x < 1.0.
    """
    ...

def atanh(x: float) -> float:
    """Return the inverse hyperbolic tangent of x.

    Raises:
        ValueError: If x <= -1.0 or x >= 1.0.
    """
    ...
def sinh(x: float) -> float: ...
def cosh(x: float) -> float: ...
def tanh(x: float) -> float: ...
def asinh(x: float) -> float: ...
def acosh(x: float) -> float: ...
def atanh(x: float) -> float: ...

# ── Geometry & advanced utils ─────────────────────────────────────────────────

def hypot_3d(x: float, y: float, z: float) -> float:
    """Return sqrt(x² + y² + z²), stable against intermediate overflow."""
    ...

def fma(x: float, y: float, z: float) -> float:
    """Return (x * y) + z with a single rounding step (fused multiply-add)."""
    ...

def copysign(x: float, y: float) -> float:
    """Return x with the sign of y."""
    ...

def nextafter(x: float, y: float) -> float:
    """Return the next representable float after x in the direction of y."""
    ...

def degrees(x: float) -> float:
    """Convert x from radians to degrees."""
    ...

def radians(x: float) -> float:
    """Convert x from degrees to radians."""
    ...

def frexp(x: float) -> tuple[float, int]:
    """Return (mantissa, exponent) such that x = mantissa * 2 ** exponent.

    mantissa is in [0.5, 1.0). Returns (0.0, 0) for x = 0.0.
    """
    ...

def ulp(x: float) -> float:
    """Return the Unit in Last Place of x — the gap to the next representable float."""
    ...

# ── Predicates ────────────────────────────────────────────────────────────────

def isfinite(x: float) -> bool:
    """Return True if x is neither NaN nor infinite."""
    ...

def isinf(x: float) -> bool:
    """Return True if x is positive or negative infinity."""
    ...

def isnan(x: float) -> bool:
    """Return True if x is NaN. Use this instead of x == float('nan')."""
    ...

def is_integer(x: float) -> bool:
    """Return True if x is finite and has no fractional part.

    Returns False for NaN and infinity, matching float.is_integer().
    """
    ...

def isclose(
    a: float,
    b: float,
    *,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
) -> bool:
    """Return True if a and b are close, mirroring math.isclose().

    Uses: |a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)
    """
    ...
def isfinite(x: float) -> bool: ...
def isinf(x: float) -> bool: ...
def isnan(x: float) -> bool: ...
def is_integer(x: float) -> bool: ...
def isclose(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool: ...

# ── Integer & bitwise ─────────────────────────────────────────────────────────

def factorial(n: int) -> int:
    """Return n! as an integer. Maximum n is 34 (u128 limit).

    Raises:
        OverflowError: If n > 34.
    """
    ...

def gcd(a: int, b: int) -> int:
    """Return the Greatest Common Divisor of a and b, always >= 0.

    Handles negative inputs. gcd(0, 0) returns 0.
    """
    ...

def lcm(a: int, b: int) -> int:
    """Return the Least Common Multiple of a and b, always >= 0.

    Raises:
        OverflowError: If the result exceeds i64 range.
    """
    ...

def is_power_of_two(n: int) -> bool:
    """Return True if n is a power of 2 (n = 2^k for k >= 0)."""
    ...

def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n.

    Raises:
        OverflowError: If n > 2^62.
    """
    ...

def is_prime(n: int) -> bool:
    """Return True if n is a prime number. Uses O(sqrt(n)) wheel factorisation."""
    ...

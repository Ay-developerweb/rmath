"""
rmath.scalar — high-performance scalar math operations backed by Rust.

This module provides Rust-accelerated scalar types and the LazyPipeline for loop fusion.
By wrapping 64-bit IEEE-754 floats in a native Rust container, we enable math operations 
that release the Python GIL and achieve significantly higher performance than pure Python 
loops.

Performance Philosophy:
    Rmath scalar operations are designed for high-throughput numeric workloads. 
    While Python's built-in `float` is highly optimized for general-purpose use, 
    `rmath.Scalar` allows for chaining operations that execute entirely in Rust, 
    avoiding the overhead of the Python interpreter for every intermediate step.

NaN and Error Policy:
    Rmath distinguishes between *operators* and *named methods*:
    
    1. Operators (+, -, *, /, **) follow IEEE-754 semantics strictly. They 
       never raise exceptions for domain errors (like division by zero or 
       sqrt of negative); they return `NaN` or `inf` to maintain 
       interoperability with Python's numeric tower.
       
    2. Named Methods (.sqrt(), .log(), .pow()) are stricter. Since they represent 
       a deliberate mathematical call, they raise `ValueError` for domain 
       errors to help catch bugs early in the pipeline.

Import style::

    import rmath.scalar as rs
    s = rs.Scalar(1.0)
    result = s.sin().cos().exp()

Note:
    Constants like `rs.pi` and `rs.e` are returned as `Scalar` objects to allow
    zero-overhead arithmetic with other Scalar instances.
"""

from typing import Optional, Union, Tuple, List, Sequence, Any

# ── Classes ──────────────────────────────────────────────────────────────────

class Scalar:
    """A high-performance, Rust-backed scalar numeric value.
    
    Wraps a 64-bit IEEE-754 float (`f64`) and exposes it to Python with a complete
    numeric protocol. Scalar objects are immutable and copy-on-operation, similar
    to Python's built-in `float`, but they allow for operations to be offloaded
    to Rust's optimized math libraries.

    Example:
        >>> from rmath import scalar as sc
        >>> x = sc.Scalar(3.14)
        >>> y = sc.Scalar(2.0)
        >>> x + y
        Scalar(5.14)
        >>> float(x)
        3.14
    """

    def __init__(self, value: float) -> None:
        """Create a new high-precision Scalar from a float.
        
        Example:
            >>> s = rs.Scalar(42.0)
            >>> s
            Scalar(42.0)
        """
        ...

    def to_python(self) -> float:
        """Convert this Scalar back into a native Python `float`.
        
        This is the explicit boundary crossing back into Python-land. 
        Equivalently, you can use `float(scalar)`.
        """
        ...

    # --- Math Methods ---
    def sqrt(self) -> "Scalar":
        """Calculate the non-negative square root.
        
        Raises:
            ValueError: If the input is negative.
            
        Example:
            >>> rs.Scalar(16.0).sqrt()
            Scalar(4.0)
            >>> rs.Scalar(-1.0).sqrt()
            Traceback (most recent call last):
                ...
            ValueError: sqrt: domain error — cannot take square root of a negative Scalar
        """
        ...

    def cbrt(self) -> "Scalar":
        """Calculate the real cube root.
        
        Defined for all real numbers (including negatives).

        Example:
            >>> rs.Scalar(27.0).cbrt()
            Scalar(3.0)
            >>> rs.Scalar(-8.0).cbrt()
            Scalar(-2.0)
        """
        ...

    def pow(self, exp: float) -> "Scalar":
        """Raise this Scalar to a real power. 
        
        Raises:
            ValueError: If the result is NaN (e.g. negative base with a fractional exponent).

        Note:
            This method is stricter than the `**` operator. While `s ** 0.5` 
            returns `NaN` for negative `s`, `s.pow(0.5)` will raise a `ValueError`.
        """
        ...

    def exp(self) -> "Scalar":
        """Natural exponential: e^x."""
        ...

    def exp2(self) -> "Scalar":
        """Base-2 exponential: 2^x."""
        ...

    def log(self, base: Optional[float] = None) -> "Scalar":
        """Natural logarithm (base e), or arbitrary base if provided.
        
        Raises:
            ValueError: If input is non-positive, or if the provided `base` 
                is non-positive or equal to 1.

        Usage:
            >>> rs.Scalar(100.0).log(10.0)
            Scalar(2.0)
            >>> rs.Scalar(rs.e).log()  # Natural log
            Scalar(1.0)
        """
        ...

    def log2(self) -> "Scalar":
        """Base-2 logarithm. Faster than log(x, 2)."""
        ...
    def log10(self) -> "Scalar": ...
    def sin(self) -> "Scalar": ...
    def cos(self) -> "Scalar": ...
    def tan(self) -> "Scalar": ...
    def asin(self) -> "Scalar": ...
    def acos(self) -> "Scalar": ...

    def atan(self) -> "Scalar":
        """Arctangent in radians. Returns a value in (-π/2, π/2)."""
        ...

    def atan2(self, x: float) -> "Scalar":
        """Two-argument arctangent of (self, x).
        
        Returns the angle in (-π, π] radians — equivalent to `atan2(y, x)` where `y=self`.
        """
        ...

    def sinh(self) -> "Scalar": ...
    def cosh(self) -> "Scalar": ...
    def tanh(self) -> "Scalar": ...
    def asinh(self) -> "Scalar": ...
    def acosh(self) -> "Scalar": ...
    def atanh(self) -> "Scalar": ...

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
        """Truncate toward zero (drop fractional part)."""
        ...

    def fract(self) -> "Scalar":
        """Return the fractional part: x - trunc(x)."""
        ...

    def abs(self) -> "Scalar":
        """Return the absolute value |x|.
        
        Note:
            This is the explicit method version of the `abs()` builtin.
        """
        ...

    def hypot(self, other: float) -> "Scalar":
        """Numerically stable square root of the sum of squares: sqrt(x² + y²)."""
        ...
    
    # --- Predicates ---
    def is_nan(self) -> bool:
        """Returns True if the value is 'Not a Number'."""
        ...

    def is_inf(self) -> bool:
        """Returns True if the value is positive or negative infinity."""
        ...

    def is_finite(self) -> bool:
        """Returns True if the value is neither NaN nor infinite."""
        ...

    # --- Utility ---
    def clamp(self, low: float, high: float) -> "Scalar":
        """Clamp the value to the range [low, high].
        
        Raises:
            ValueError: If `low > high`.
        """
        ...

    def signum(self) -> "Scalar":
        """Return the sign of the value: -1.0, 0.0, or 1.0.
        
        NaN input returns NaN.
        """
        ...

    def lerp(self, other: float, t: float) -> "Scalar":
        """Linear interpolation between self and other: `self + t * (other - self)`.
        
        `t = 0.0` returns `self`, `t = 1.0` returns `other`. 
        `t` is not clamped — extrapolation is allowed.
        """
        ...

    # --- Protocol (Numeric) ---
    def __add__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Addition operator: self + other.
        
        Supports addition with other Scalar instances, ints, or floats.
        """
        ...
    def __radd__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Reflected addition: other + self."""
        ...
    def __sub__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Subtraction operator: self - other."""
        ...
    def __rsub__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Reflected subtraction: other - self."""
        ...
    def __mul__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Multiplication operator: self * other."""
        ...
    def __rmul__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Reflected multiplication: other * self."""
        ...
    def __truediv__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """True division operator: self / other.
        
        Raises:
            ZeroDivisionError: If other is 0.
        """
        ...
    def __rtruediv__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Reflected true division: other / self."""
        ...
    def __floordiv__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Floor division operator: self // other."""
        ...
    def __rfloordiv__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Reflected floor division: other // self."""
        ...
    def __mod__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Modulo operator: self % other."""
        ...
    def __rmod__(self, other: Union[float, "Scalar"]) -> "Scalar":
        """Reflected modulo: other % self."""
        ...
    def __pow__(self, other: Union[float, "Scalar"], modulo: Optional[Any] = None) -> "Scalar":
        """Power operator: self ** other.
        
        Matches IEEE-754 semantics: returns NaN for domain errors (e.g. -1 ** 0.5)
        rather than raising. For strict error checking, use `.pow()`.
        """
        ...
    def __rpow__(self, other: Union[float, "Scalar"], modulo: Optional[Any] = None) -> "Scalar":
        """Reflected power operator: other ** self."""
        ...
    def __neg__(self) -> "Scalar": ...
    def __abs__(self) -> "Scalar": ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __hash__(self) -> int: ...


class Complex:
    """A high-performance complex number (re + im*j) backed by Rust.
    
    Distinct from `Scalar` by design — users working purely in ℝ never pay the
    branch-prediction or cognitive cost of complex arithmetic paths.
    
    Provides specialized kernels for complex arithmetic, including polar
    conversions and complex transcendental functions (exp, log, sqrt).
    """
    def __init__(self, re: float, im: float) -> None:
        """Create a new complex number: `re + im*j`.
        
        Example:
            >>> c = rs.Complex(3.0, 4.0)
            >>> c
            (3.000000+4.000000j)
        """
        ...

    @staticmethod
    def from_polar(r: float, theta: float) -> "Complex":
        """Construct a complex number from polar coordinates (r, θ).
        
        `r` is the modulus (must be >= 0), `theta` is the argument in radians.
        
        Usage:
            >>> c = rs.Complex.from_polar(1.0, rs.pi / 2)
            >>> c.im
            1.0
        """
        ...

    @property
    def re(self) -> float:
        """The real component of the complex number."""
        ...

    @property
    def im(self) -> float:
        """The imaginary component of the complex number."""
        ...

    def abs(self) -> float:
        """The modulus (magnitude) |z| = √(re² + im²).
        
        Usage:
            >>> rs.Complex(3.0, 4.0).abs()
            5.0
        """
        ...

    def arg(self) -> float:
        """The argument (phase angle) in radians, in (-π, π]."""
        ...

    def conjugate(self) -> "Complex":
        """Return the complex conjugate (re - im*j)."""
        ...

    def to_polar(self) -> Tuple[float, float]:
        """Convert to polar form: returns (modulus, argument)."""
        ...

    def exp(self) -> "Complex":
        """Complex exponential: e^z."""
        ...
    def log(self) -> "Complex":
        """Complex natural logarithm: ln(z).
        
        Raises:
            ValueError: If z is zero.
        """
        ...
    def sqrt(self) -> "Complex":
        """Complex square root (principal branch)."""
        ...
    def pow(self, other: Union[float, "Complex", Scalar]) -> "Complex":
        """Complex power: z^other.
        
        Raises:
            ValueError: If base is zero and exponent has non-positive real part.
        """
        ...
    def sin(self) -> "Complex":
        """Complex sine."""
        ...
    def cos(self) -> "Complex":
        """Complex cosine."""
        ...
    
    def __abs__(self) -> float:
        """Absolute value (modulus)."""
        ...
    def __neg__(self) -> "Complex":
        """Unary negation: -z."""
        ...
    def __add__(self, other: Union[float, "Complex", Scalar]) -> "Complex":
        """Addition operator: self + other."""
        ...
    def __mul__(self, other: Union[float, "Complex", Scalar]) -> "Complex":
        """Multiplication operator: self * other."""
        ...
    def __repr__(self) -> str: ...


class LazyPipeline:
    """A lazy, parallel evaluation pipeline for scalar operations.
    
    Note:
        Do not instantiate this class directly. Instead, use one of the
        factory entry points:
        
        - rs.from_list()
        - rs.loop_range()
        - rs.linspace()
        - rs.zeros()

    The LazyPipeline is the primary engine for high-performance batch processing
    of scalar data. It captures a 'recipe' of operations and executes them in 
    a single parallel pass using Rayon.

    Evaluation is deferred until a terminal operation (like `.to_tuple()`, 
    `.sum()`, or `.to_vector()`) is called. This allows for loop fusion 
    and SIMD optimizations.

    Usage:
        >>> import rmath.scalar as rs
        >>> # The pipeline is returned by the factory 'from_list'
        >>> pipe = rs.from_list([1.0, 2.0, 3.0])
        >>> result = pipe.sin().cos().add(1.0).to_tuple()
    """
    def to_vector(self) -> Any:
        """Execute the pipeline and return the result as an `rmath.vector.Vector`."""
        ...

    def to_tuple(self) -> Tuple[float, ...]:
        """Execute the pipeline and return the result as a native Python tuple.
        
        This is the preferred way to iterate over pipeline results in a
        Python `for` loop. The tuple contains plain Python `float` objects,
        which are cheaper to construct and iterate over than `Scalar` objects.
        """
        ...

    def sum(self) -> float:
        """Perform a single-pass parallel reduction to calculate the sum.
        
        Returns 0.0 for an empty pipeline.
        """
        ...

    def mean(self) -> float:
        """Arithmetic mean.
        
        Uses Welford's online algorithm for numerical stability.
        Returns `NaN` for an empty pipeline.
        """
        ...

    def var(self) -> float:
        """Population variance."""
        ...

    def std(self) -> float:
        """Standard deviation."""
        ...

    def max(self) -> float:
        """Find the maximum value in the pipeline."""
        ...

    def min(self) -> float:
        """Find the minimum value in the pipeline."""
        ...

    # --- Chaining (Mapping) ---
    def sin(self) -> "LazyPipeline": ...
    def cos(self) -> "LazyPipeline": ...
    def sqrt(self) -> "LazyPipeline": ...
    def abs(self) -> "LazyPipeline": ...
    def exp(self) -> "LazyPipeline": ...
    
    def add(self, val: float) -> "LazyPipeline":
        """Add a scalar value to every element in the pipeline."""
        ...

    def mul(self, val: float) -> "LazyPipeline":
        """Multiply every element in the pipeline by a scalar value."""
        ...

    def sub(self, val: float) -> "LazyPipeline":
        """Subtract a scalar value from every element in the pipeline."""
        ...

    def div(self, val: float) -> "LazyPipeline":
        """Divide every element in the pipeline by a scalar value.
        
        Raises:
            ZeroDivisionError: If `val == 0`.
        """
        ...

    # --- Chaining (Filtering) ---
    def filter_gt(self, val: float) -> "LazyPipeline":
        """Only keep elements greater than `val`."""
        ...
    def filter_lt(self, val: float) -> "LazyPipeline":
        """Only keep elements less than `val`."""
        ...
    def filter_finite(self) -> "LazyPipeline":
        """Only keep elements that are not NaN or Infinity."""
        ...

    def as_(self, target: str) -> "LazyPipeline":
        """Cast the pipeline to a specific target type hint."""
        ...

    def __len__(self) -> int:
        """Returns the source length before filters.
        
        After filters, the actual count is unknown until evaluation.
        """
        ...
    def __repr__(self) -> str: ...

# ── Functions ────────────────────────────────────────────────────────────────

def from_list(data: Sequence[float]) -> LazyPipeline:
    """Create a new LazyPipeline from a Python list of floats.
    
    Usage:
        >>> pipe = rs.from_list([1.0, 2.0, 3.0])
        >>> pipe.sin().to_tuple()
    """
    ...

def loop_range(start: float, stop: Optional[float] = None, step: float = 1.0) -> LazyPipeline:
    """Create a new LazyPipeline from a numeric range.
    
    Returns values from `start` up to (but not including) `stop`.
    If `stop` is not provided, `start` is used as the stop value and 
    0.0 is used as the start.

    Raises:
        ValueError: If `step` is zero or has the wrong sign.

    Usage:
        >>> # Generate 1 million values and sum them in parallel
        >>> rs.loop_range(0.0, 1_000_000.0).sum()
    """
    ...

def zeros(n: int) -> LazyPipeline:
    """Create a new LazyPipeline of `n` zeros."""
    ...

def linspace(start: float, stop: float, num: int) -> LazyPipeline:
    """Create a new LazyPipeline with `num` evenly spaced values.
    
    The values range from `start` to `stop` (inclusive).

    Raises:
        ValueError: If `num < 1`.

    Usage:
        >>> rs.linspace(0.0, 1.0, 5).to_tuple()
        (0.0, 0.25, 0.5, 0.75, 1.0)
    """
    ...

# ── Constants ─────────────────────────────────────────────────────────────────

pi: Scalar
"""Mathematical constant π (3.14159...)."""

e: Scalar
"""Euler's number (2.71828...)."""

tau: Scalar
"""Mathematical constant τ (2π)."""

inf: Scalar
"""Positive infinity."""

nan: Scalar
"""'Not a Number' value."""

sqrt2: Scalar
"""Square root of 2."""

ln2: Scalar
"""Natural logarithm of 2."""

ln10: Scalar
"""Natural logarithm of 10."""

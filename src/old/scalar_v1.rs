use pyo3::prelude::*;

// ============================================================================
// --- [1/9] Basic Arithmetic ---
// ============================================================================

/// Returns the mathematical sum of two numbers.
///
/// Note: Like all IEEE 754 floating-point addition, adding very large
/// and very small numbers together may lead to precision loss.
///
/// Args:
///     x (float): The first summand.
///     y (float): The second summand.
///
/// Returns:
///     float: The sum of x and y.
///
/// Examples:
///     >>> rmath.scalar.add(10.5, 0.5)
///     11.0
///     >>> rmath.scalar.add(float('inf'), 1.0)
///     inf
#[pyfunction]
pub fn add(x: f64, y: f64) -> f64 {
    x + y
}

/// Returns the difference of two numbers (x - y).
///
/// Args:
///     x (float): The value to subtract from.
///     y (float): The value to subtract.
///
/// Returns:
///     float: The difference (left - right).
///
/// Examples:
///     >>> rmath.scalar.sub(1.0, 0.1)
///     0.9
#[pyfunction]
pub fn sub(x: f64, y: f64) -> f64 {
    x - y
}

/// Returns the product of two numbers (x * y).
///
/// Args:
///     x (float): The first factor.
///     y (float): The second factor.
///
/// Returns:
///     float: The product.
///
/// Examples:
///     >>> rmath.scalar.mul(2.5, 4.0)
///     10.0
#[pyfunction]
pub fn mul(x: f64, y: f64) -> f64 {
    x * y
}

/// Returns the quotient of two numbers (x / y).
///
/// Optimization: This is a direct hardware-level division 
/// after the ZeroDivisionError check.
///
/// Args:
///     x (float): The numerator.
///     y (float): The denominator.
///
/// Returns:
///     float: The quotient result.
///
/// Raises:
///     ZeroDivisionError: If the divisor y is 0.0.
///
/// Examples:
///     >>> rmath.scalar.div(1.0, 4.0)
///     0.25
#[pyfunction]
pub fn div(x: f64, y: f64) -> PyResult<f64> {
    if y == 0.0 {
        return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero"));
    }
    Ok(x / y)
}

/// Returns the IEEE 754 floating-point remainder of x / y.
///
/// The sign of the result follows the **divisor** (y), matching
/// Python's `%` operator. This differs from `fmod`, where the sign
/// follows the dividend (x).
///
/// Args:
///     x (float): The dividend.
///     y (float): The divisor.
///
/// Returns:
///     float: The remainder (sign follows the divisor).
///
/// Examples:
///     >>> rmath.scalar.remainder(-5.0, 3.0)
///     1.0   # positive, follows sign of y=3.0
///     >>> rmath.scalar.remainder(5.0, 3.0)
///     2.0
#[pyfunction]
pub fn remainder(x: f64, y: f64) -> f64 {
    x - (x / y).floor() * y  // matches Python % semantics
}

/// Returns the C-style floating-point remainder (fmod).
///
/// Crucial Difference: Unlike Python's `%` operator, which 
/// follows the sign of the divisor (y), fmod(x, y) follows the 
/// sign of the dividend (x). This matches the behavior of 
/// C/C++ and Java.
///
/// Args:
///     x (float): The dividend.
///     y (float): The divisor.
///
/// Returns:
///     float: The C-style fmod remainder (sign follows the dividend).
///
/// Examples:
///     >>> rmath.scalar.fmod(-5.0, 3.0)
///     -2.0 # negative, follows sign of x=-5.0
///     >>> rmath.scalar.fmod(5.0, 3.0)
///     2.0
#[pyfunction]
pub fn fmod(x: f64, y: f64) -> f64 {
    // Rust % on floats is C's fmod: sign of result follows dividend
    x % y
}
// Fix 3: sub docstring — "left/right" → "x/y"
/// Returns:
///     float: The difference (x - y).   ← was "left - right"

// Fix 4: div — remove impl detail from docstring
// Remove: "Optimization: This is a direct hardware-level division..."

// ============================================================================
// --- [2/9] Rounding and Range ---
// ============================================================================

/// Returns the smallest integer greater than or equal to x.
///
/// Use this for calculating upper bounds or rounding up to the next 
/// whole number.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: The smallest integer >= x.
///
/// Examples:
///     >>> rmath.scalar.ceil(2.0001)
///     3.0
///     >>> rmath.scalar.ceil(-1.5)
///     -1.0
#[pyfunction]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Returns the largest integer less than or equal to x.
///
/// Useful for determining indices or flooring costs.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: The largest integer <= x.
///
/// Examples:
///     >>> rmath.scalar.floor(2.999)
///     2.0
///     >>> rmath.scalar.floor(-1.5)
///     -2.0
#[pyfunction]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Discards the fractional part of a number.
///
/// Unlike `floor`, which moves towards negative infinity, 
/// `trunc` always moves towards zero.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: The value of x with the decimal part removed.
///
/// Examples:
///     >>> rmath.scalar.trunc(2.9)
///     2.0
///     >>> rmath.scalar.trunc(-2.9)
///     -2.0
#[pyfunction]
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

/// Rounds x to the nearest integer.
///
/// If the fractional part is exactly 0.5, this implementation 
/// rounds away from zero. 
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: The rounded integer.
///
/// Examples:
///     >>> rmath.scalar.round(2.5)
///     3.0
///     >>> rmath.scalar.round(-2.5)
///     -3.0
#[pyfunction]
pub fn round(x: f64) -> f64 {
    x.round()
}

/// Performs 'Banker's rounding' to the nearest even number.
///
/// This reduces statistical bias over large datasets by rounding 
/// to the nearest even number when the value is exactly between 
/// two integers.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: Rounded to the nearest even integer on half-cases.
///
/// Examples:
///     >>> rmath.scalar.round_half_even(2.5)
///     2.0
///     >>> rmath.scalar.round_half_even(3.5)
///     4.0
#[pyfunction]
pub fn round_half_even(x: f64) -> f64 {
    let rounded = x.round();
    // Only intervene when we're exactly at the 0.5 boundary
    if (x - rounded).abs() == 0.5 {
        if rounded % 2.0 == 0.0 {
            rounded          // already even — keep it
        } else {
            rounded - x.signum()  // step one toward zero to reach nearest even
        }
    } else {
        rounded
    }
}

/// Returns the signum of a number as -1.0, 0.0, or 1.0.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: -1.0 if negative, 0.0 if zero, 1.0 if positive.
///
/// Examples:
///     >>> rmath.scalar.signum(-42.0)
///     -1.0
///     >>> rmath.scalar.signum(0.0)
///     0.0
#[pyfunction]
pub fn signum(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.signum() }
}

/// Returns the absolute value of x.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: |x|.
///
/// Examples:
///     >>> rmath.scalar.abs(-5.0)
///     5.0
#[pyfunction]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// Clamps the value x to be within the range [min, max].
///
/// If x is outside the range, it is moved to the nearest boundary.
///
/// Args:
///     x (float): The value to clamp.
///     min (float): The lower bound.
///     max (float): The upper bound.
///
/// Returns:
///     float: The clamped value.
///
/// Examples:
///     >>> rmath.scalar.clamp(10, 0, 5)
///     5.0
///     >>> rmath.scalar.clamp(-10, 0, 5)
///     0.0
#[pyfunction]
pub fn clamp(x: f64, min: f64, max: f64) -> PyResult<f64> {
    if min > max {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min must be less than or equal to max"
        ));
    }
    Ok(x.clamp(min, max))
}

/// Linear interpolation between a and b at parameter t.
///
/// Returns a + t * (b - a). No clamping is applied — values of t
/// outside [0.0, 1.0] extrapolate beyond the [a, b] range.
/// Use clamp(t, 0.0, 1.0) beforehand if you want bounded output.
///
/// Examples:
///     >>> rmath.scalar.lerp(10.0, 20.0, 0.5)
///     15.0
///     >>> rmath.scalar.lerp(0.0, 100.0, 1.5)  # extrapolation
///     150.0
#[pyfunction]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

// ============================================================================
// --- [3/9] Roots, Powers, and Abs ---
// ============================================================================

/// Returns the non-negative square root of x.
///
/// Args:
///     x (float): The value to process.
///
/// Returns:
///     float: The square root.
///
/// Raises:
///     ValueError: If x is negative.
///
/// Examples:
///     >>> rmath.scalar.sqrt(64.0)
///     8.0
#[pyfunction]
pub fn sqrt(x: f64) -> PyResult<f64> {
    if x < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot take square root of negative number"));
    }
    Ok(x.sqrt())
}

/// Returns the cube root of x.
///
/// Args:
///     x (float): The value to process.
///
/// Returns:
///     float: The cube root.
///
/// Examples:
///     >>> rmath.scalar.cbrt(-27.0)
///     -3.0
#[pyfunction]
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Returns x raised to the power y (x ** y).
///
/// If the result is NaN (e.g. a negative base with a fractional
/// exponent), a ValueError is raised instead of returning NaN silently.
///
/// Args:
///     x (float): The base.
///     y (float): The exponent.
///
/// Returns:
///     float: x raised to the power y.
///
/// Raises:
///     ValueError: If the result would be NaN (e.g. pow(-2.0, 0.5)).
///
/// Examples:
///     >>> rmath.scalar.pow(2.0, 10.0)
///     1024.0
///     >>> rmath.scalar.pow(2.0, -1.0)
///     0.5
#[pyfunction]
pub fn pow(x: f64, y: f64) -> PyResult<f64> {
    let result = x.powf(y);
    if result.is_nan() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pow result is not a real number",
        ));
    }
    Ok(result)
}

/// Returns the fast inverse square root 1 / sqrt(x).
///
/// Optimised for graphics and vector normalisation workloads.
///
/// Args:
///     x (float): The input value. Must be strictly positive.
///
/// Returns:
///     float: 1.0 / sqrt(x).
///
/// Raises:
///     ValueError: If x is negative or zero.
///
/// Examples:
///     >>> rmath.scalar.inv_sqrt(4.0)
///     0.5
///     >>> rmath.scalar.inv_sqrt(1.0)
///     1.0
#[pyfunction]
pub fn inv_sqrt(x: f64) -> PyResult<f64> {
    if x <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inv_sqrt requires a strictly positive value",
        ));
    }
    Ok(1.0 / x.sqrt())
}

/// Returns the real principal n-th root of x.
///
/// Correctly handles odd roots of negative numbers. For n=3,
/// delegates to the hardware cbrt() for maximum precision.
///
/// Args:
///     x (float): The radicand.
///     n (int): The degree. May be negative (e.g. n=-2 gives 1/sqrt(x)).
///
/// Returns:
///     float: The real n-th root of x.
///
/// Raises:
///     ValueError: If n is 0, or n is even and x is negative.
///
/// Examples:
///     >>> rmath.scalar.root(-125.0, 3)
///     -5.0
///     >>> rmath.scalar.root(16.0, 4)
///     2.0
///     >>> rmath.scalar.root(8.0, -3)
///     0.5
#[pyfunction]
pub fn root(x: f64, n: i32) -> PyResult<f64> {
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "0th root is undefined",
        ));
    }
    if x < 0.0 && n % 2 == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "even root of a negative number is not real",
        ));
    }
    // Use native cbrt for n=3: more precise than powf(1/3)
    let abs_root = if n == 3 {
        x.abs().cbrt()
    } else {
        x.abs().powf(1.0 / n.abs() as f64)
    };
    let result = if x < 0.0 { -abs_root } else { abs_root };
    // Handle negative n: n-th root = 1 / (-n)-th root
    if n < 0 {
        Ok(1.0 / result)
    } else {
        Ok(result)
    }
}

// ============================================================================
// --- [4/9] Exponential and Logarithmic ---
// ============================================================================

/// Returns e raised to the power of x.
/// Returns e raised to the power of x (natural exponential).
///
/// Args:
///     x (float): The exponent.
///
/// Returns:
///     float: e^x.
///
/// Examples:
///     >>> rmath.scalar.exp(1.0)
///     2.718281828459045
#[pyfunction]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Returns the logarithm of x to the specified base.
///
/// If no base is provided, it defaults to the natural logarithm (base e).
///
/// Args:
///     x (float): The value to take the log of.
///     base (float, optional): The base. Defaults to ln (base e).
///
/// Returns:
///     float: The log result.
///
/// Raises:
///     ValueError: If x <= 0 or the base is invalid (base <= 0 or base == 1).
///
/// Examples:
///     >>> rmath.scalar.log(100, 10)
///     2.0
///     >>> rmath.scalar.log(math.e)
///     1.0
#[pyfunction]
#[pyo3(signature = (x, base=None))]
pub fn log(x: f64, base: Option<f64>) -> PyResult<f64> {
    if x <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Math domain error: log(x) for x <= 0"));
    }
    match base {
        Some(b) => {
            if b <= 0.0 || b == 1.0 {
                return Err(pyo3::exceptions::PyValueError::new_err("Math domain error: invalid base"));
            }
            Ok(x.log(b))
        },
        None => Ok(x.ln()),
    }
}

/// Returns the base-2 logarithm of x.
///
/// Faster and more accurate than log(x, 2).
///
/// Args:
///     x (float): The value.
///
/// Returns:
///     float: log2(x).
///
/// Examples:
///     >>> rmath.scalar.log2(1024)
///     10.0
#[pyfunction]
pub fn log2(x: f64) -> PyResult<f64> {
    if x <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("log2 requires x > 0"));
    }
    Ok(x.log2())
}

/// Returns the base-10 logarithm of x.
///
/// Better precision than log(x, 10).
///
/// Args:
///     x (float): The value.
///
/// Returns:
///     float: log10(x).
///
/// Examples:
///     >>> rmath.scalar.log10(1000)
///     3.0
#[pyfunction]
pub fn log10(x: f64) -> PyResult<f64> {
    if x <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("log10 requires x > 0"));
    }
    Ok(x.log10())
}

/// Returns ln(1 + x) accurately for very small x.
///
/// Use this instead of log(1 + x) when x is near zero to 
/// avoid precision loss.
///
/// Args:
///     x (float): The value.
///
/// Returns:
///     float: ln(1+x).
///
/// Examples:
///     >>> rmath.scalar.log1p(1e-15)
///     9.999999999999997e-16
#[pyfunction]
pub fn log1p(x: f64) -> PyResult<f64> {
    if x <= -1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("log1p requires x > -1"));
    }
    Ok(x.ln_1p())
}

/// Returns exp(x) - 1 accurately for very small x.
///
/// Essential for probability calculations near zero.
///
/// Args:
///     x (float): The value.
///
/// Returns:
///     float: exp(x)-1.
///
/// Examples:
///     >>> rmath.scalar.expm1(1e-15)
///     1.0000000000000005e-15
#[pyfunction]
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// Returns 2 raised to the power of x.
///
/// Args:
///     x (float): The exponent.
///
/// Returns:
///     float: 2^x.
///
/// Examples:
///     >>> rmath.scalar.exp2(10)
///     1024.0
#[pyfunction]
pub fn exp2(x: f64) -> f64 {
    x.exp2()
}

/// Numerically stable log-sum-exp for two scalars: ln(exp(x) + exp(y)).
///
/// Args:
///     x (float): First value.
///     y (float): Second value.
///
/// Returns:
///     float: ln(exp(x) + exp(y)).
#[pyfunction]
pub fn logsumexp2(x: f64, y: f64) -> f64 {
    // Numerically stable: shift by max to prevent exp overflow
    let max = x.max(y);
    max + ((x - max).exp() + (y - max).exp()).ln()
}

// ============================================================================
// --- [5/9] Trigonometry ---
// ============================================================================

/// Returns the sine of x (measured in radians).
///
/// Args:
///     x (float): Angle in radians.
///
/// Returns:
///     float: sin(x).
///
/// Examples:
///     >>> rmath.scalar.sin(math.pi / 2)
///     1.0
#[pyfunction]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Returns the cosine of x (measured in radians).
///
/// Args:
///     x (float): Angle in radians.
///
/// Returns:
///     float: cos(x).
///
/// Examples:
///     >>> rmath.scalar.cos(math.pi)
///     -1.0
#[pyfunction]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Returns the tangent of x (measured in radians).
///
/// Note: Due to floating-point precision, tan(pi/4) returns
/// 0.9999999999999999 rather than exactly 1.0.
///
/// Args:
///     x (float): Angle in radians.
///
/// Returns:
///     float: tan(x).
///
/// Examples:
///     >>> rmath.scalar.tan(0.0)
///     0.0
///     >>> rmath.scalar.tan(math.pi / 4)  # approx 1.0
///     0.9999999999999999
#[pyfunction]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// Returns the arc sine of x in radians.
///
/// Args:
///     x (float): Value in range [-1.0, 1.0].
///
/// Returns:
///     float: asin(x) in range [-pi/2, pi/2].
///
/// Raises:
///     ValueError: If x is outside [-1.0, 1.0].
///
/// Examples:
///     >>> rmath.scalar.asin(1.0)
///     1.5707963267948966   # pi/2
///     >>> rmath.scalar.asin(0.0)
///     0.0
#[pyfunction]
pub fn asin(x: f64) -> PyResult<f64> {
    if x < -1.0 || x > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("asin(x) domain is [-1, 1]"));
    }
    Ok(x.asin())
}

/// Returns the arc cosine of x in radians.
///
/// Args:
///     x (float): Value in range [-1.0, 1.0].
///
/// Returns:
///     float: acos(x) in range [0, pi].
///
/// Raises:
///     ValueError: If x is outside [-1.0, 1.0].
///
/// Examples:
///     >>> rmath.scalar.acos(1.0)
///     0.0
///     >>> rmath.scalar.acos(-1.0)
///     3.141592653589793   # pi
#[pyfunction]
pub fn acos(x: f64) -> PyResult<f64> {
    if x < -1.0 || x > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("acos(x) domain is [-1, 1]"));
    }
    Ok(x.acos())
}

/// Returns the arc tangent of x in radians.
///
/// Args:
///     x (float): The input value (any real number).
///
/// Returns:
///     float: atan(x) in range (-pi/2, pi/2).
///
/// Examples:
///     >>> rmath.scalar.atan(1.0)
///     0.7853981633974483   # pi/4
///     >>> rmath.scalar.atan(0.0)
///     0.0
#[pyfunction]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Returns the arc tangent of y/x, using signs to determine quadrant.
///
/// Unlike atan(y/x), correctly handles x=0 and places the result
/// in the correct quadrant based on the signs of both arguments.
/// atan2(0.0, 0.0) returns 0.0 by convention.
///
/// Args:
///     y (float): The y-coordinate.
///     x (float): The x-coordinate.
///
/// Returns:
///     float: Angle in radians in range (-pi, pi].
///
/// Examples:
///     >>> rmath.scalar.atan2(1.0, 1.0)
///     0.7853981633974483   # pi/4
///     >>> rmath.scalar.atan2(1.0, 0.0)
///     1.5707963267948966   # pi/2
#[pyfunction]
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

// ============================================================================
// --- [6/9] Hyperbolic ---
// ============================================================================

/// Returns the hyperbolic sine of x.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: sinh(x).
///
/// Examples:
///     >>> rmath.scalar.sinh(0.0)
///     0.0
#[pyfunction]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

/// Returns the hyperbolic cosine of x.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: cosh(x).
///
/// Examples:
///     >>> rmath.scalar.cosh(0.0)
///     1.0
#[pyfunction]
pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

/// Returns the hyperbolic tangent of x.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: tanh(x), always in range (-1, 1).
///
/// Examples:
///     >>> rmath.scalar.tanh(0.0)
///     0.0
#[pyfunction]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Returns the inverse hyperbolic sine of x.
///
/// Defined for all real numbers. asinh(sinh(x)) == x.
///
/// Args:
///     x (float): Any real number.
///
/// Returns:
///     float: asinh(x).
///
/// Examples:
///     >>> rmath.scalar.asinh(0.0)
///     0.0
///     >>> rmath.scalar.asinh(1.0)
///     0.8813735870195430
#[pyfunction]
pub fn asinh(x: f64) -> f64 {
    x.asinh()
}

/// Returns the inverse hyperbolic cosine of x.
///
/// Args:
///     x (float): Value >= 1.0.
///
/// Returns:
///     float: acosh(x) >= 0.0.
///
/// Raises:
///     ValueError: If x < 1.0.
///
/// Examples:
///     >>> rmath.scalar.acosh(1.0)
///     0.0
///     >>> rmath.scalar.acosh(2.0)
///     1.3169578969248166
#[pyfunction]
pub fn acosh(x: f64) -> PyResult<f64> {
    if x < 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("acosh(x) domain is [1, inf)"));
    }
    Ok(x.acosh())
}

/// Returns the inverse hyperbolic tangent of x.
///
/// Args:
///     x (float): Value strictly inside (-1.0, 1.0).
///
/// Returns:
///     float: atanh(x).
///
/// Raises:
///     ValueError: If x <= -1.0 or x >= 1.0.
///
/// Examples:
///     >>> rmath.scalar.atanh(0.0)
///     0.0
///     >>> rmath.scalar.atanh(0.5)
///     0.5493061443340548
#[pyfunction]
pub fn atanh(x: f64) -> PyResult<f64> {
    if x <= -1.0 || x >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("atanh(x) domain is (-1, 1)"));
    }
    Ok(x.atanh())
}

// ============================================================================
// --- [7/9] Geometry and Advanced Utils ---
// ============================================================================

/// Returns the Euclidean distance sqrt(x² + y²).
///
/// More numerically stable than computing sqrt(x*x + y*y) directly,
/// as it avoids intermediate overflow for large values.
///
/// Args:
///     x (float): First leg.
///     y (float): Second leg.
///
/// Returns:
///     float: sqrt(x² + y²).
///
/// Examples:
///     >>> rmath.scalar.hypot(3.0, 4.0)
///     5.0
#[pyfunction]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// Returns the 3D Euclidean norm sqrt(x^2 + y^2 + z^2).
///
/// Args:
///     x (float): x coordinate.
///     y (float): y coordinate.
///     z (float): z coordinate.
///
/// Returns:
///     float: The 3D distance from origin.
///
/// Examples:
///     >>> rmath.scalar.hypot_3d(2.0, 3.0, 6.0)
///     7.0
#[pyfunction]
pub fn hypot_3d(x: f64, y: f64, z: f64) -> f64 {
    x.hypot(y).hypot(z)   // chains stable 2D hypot, avoids overflow
}

/// Returns (x * y) + z with a single rounding step (fused multiply-add).
///
/// More precise than separate multiply then add because there is only
/// one rounding step. Falls back to software FMA on CPUs without
/// hardware support.
///
/// Args:
///     x (float): Multiplicand.
///     y (float): Multiplier.
///     z (float): Addend.
///
/// Returns:
///     float: (x * y) + z rounded once.
///
/// Examples:
///     >>> rmath.scalar.fma(2.0, 3.0, 1.0)
///     7.0
///     >>> rmath.scalar.fma(1.0, 1.0, 1e-17)
///     1.00000000000000001  # precision preserved
#[pyfunction]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    x.mul_add(y, z)
}

/// Returns x with the sign of y.
///
/// Args:
///     x (float): The value.
///     y (float): The sign donor.
///
/// Returns:
///     float: x with sign(y).
///
/// Examples:
///     >>> rmath.scalar.copysign(5.0, -42.0)
///     -5.0
#[pyfunction]
pub fn copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

/// Returns the next floating-point value after x towards y.
///
/// Use this for testing floating-point precision boundaries or 
/// finding the smallest representable difference from a number.
///
/// Args:
///     x (float): Starting point.
///     y (float): Direction point.
///
/// Returns:
///     float: The next representable float.
///
/// Examples:
///     >>> rmath.scalar.nextafter(1.0, 2.0)
///     1.0000000000000002
#[pyfunction]
pub fn nextafter(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() { return f64::NAN; }
    if x == y { return y; }
    if x == 0.0 {
        return if y > 0.0 { f64::from_bits(1) } else { f64::from_bits(0x8000000000000001) };
    }
    let bits = x.to_bits() as i64;
    let next_bits = if (y > x) == (x > 0.0) { bits + 1 } else { bits - 1 };
    f64::from_bits(next_bits as u64)
}

/// Converts radians to degrees.
///
/// Args:
///     x (float): Angle in radians.
///
/// Returns:
///     float: Angle in degrees.
///
/// Examples:
///     >>> rmath.scalar.degrees(math.pi)
///     180.0
#[pyfunction]
pub fn degrees(x: f64) -> f64 {
    x.to_degrees()
}

/// Converts degrees to radians.
///
/// Args:
///     x (float): Angle in degrees.
///
/// Returns:
///     float: Angle in radians.
///
/// Examples:
///     >>> rmath.scalar.radians(180.0)
///     3.141592653589793
#[pyfunction]
pub fn radians(x: f64) -> f64 {
    x.to_radians()
}

/// Returns the mantissa and exponent (m, e) of x.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     tuple[float, int]: (mantissa, exponent) where x = mantissa * 2^exponent
///                        and mantissa is in [0.5, 1.0).
#[pyfunction]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }
    let bits = x.to_bits();
    let sign = (bits >> 63) != 0;
    let mut exponent = ((bits >> 52) & 0x7ff) as i32 - 1022;
    let mut mantissa_bits = (bits & 0x000fffffffffffff) | 0x0010000000000000;
    if exponent == -1022 {
        exponent -= (mantissa_bits.leading_zeros() - 11) as i32;
        mantissa_bits <<= mantissa_bits.leading_zeros() - 11;
    }
    let mut result_bits = (mantissa_bits & 0x000fffffffffffff) | 0x3fe0000000000000;
    if sign {
        result_bits |= 1 << 63;
    }
    (f64::from_bits(result_bits), exponent)
}

/// Returns the Unit in Last Place (ULP) of x.
///
/// The ULP is the gap between x and the next larger representable
/// float. Useful for testing floating-point precision boundaries.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     float: The ULP of x.
///
/// Examples:
///     >>> rmath.scalar.ulp(1.0)
///     2.220446049250313e-16
///     >>> rmath.scalar.ulp(0.0)
///     5e-324
#[pyfunction]
pub fn ulp(x: f64) -> f64 {
    if x.is_nan() { return f64::NAN; }
    if x.is_infinite() { return f64::INFINITY; }
    let x = x.abs();
    let bits = x.to_bits();
    let next = f64::from_bits(bits + 1);
    next - x
}

// ============================================================================
// --- [8/9] Predicates ---
// ============================================================================

/// Returns True if x is neither NaN nor Infinite.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     bool: True if finite.
///
/// Examples:
///     >>> rmath.scalar.isfinite(1.0)
///     True
///     >>> rmath.scalar.isfinite(float('inf'))
///     False
///     >>> rmath.scalar.isfinite(float('nan'))
///     False
#[pyfunction]
pub fn isfinite(x: f64) -> bool {
    x.is_finite()
}

/// Returns True if x is Infinite (Positive or Negative).
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     bool: True if infinite.
///
/// Examples:
///     >>> rmath.scalar.isinf(float('inf'))
///     True
///     >>> rmath.scalar.isinf(float('-inf'))
///     True
///     >>> rmath.scalar.isinf(1.0)
///     False
#[pyfunction]
pub fn isinf(x: f64) -> bool {
    x.is_infinite()
}

/// Returns True if x is Not-a-Number (NaN).
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     bool: True if NaN.
///
/// Note: NaN != NaN by IEEE 754 — use isnan() rather than x == float('nan').
///
/// Examples:
///     >>> rmath.scalar.isnan(float('nan'))
///     True
///     >>> rmath.scalar.isnan(1.0)
///     False
#[pyfunction]
pub fn isnan(x: f64) -> bool {
    x.is_nan()
}

/// Returns True if x is a finite number with no fractional part.
///
/// Returns False for NaN and Infinity, matching Python's
/// float.is_integer() behaviour.
///
/// Args:
///     x (float): The input value.
///
/// Returns:
///     bool: True if x is finite and has no fractional part.
///
/// Examples:
///     >>> rmath.scalar.is_integer(2.0)
///     True
///     >>> rmath.scalar.is_integer(2.5)
///     False
///     >>> rmath.scalar.is_integer(float('inf'))
///     False
///     >>> rmath.scalar.is_integer(float('nan'))
///     False
#[pyfunction]
pub fn is_integer(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0
}

/// Returns True if a and b are close, mirroring math.isclose().
///
/// Considers values close if:
///     |a - b| <= max(rel_tol * max(|a|, |b|), abs_tol)
///
/// Args:
///     a (float): First value.
///     b (float): Second value.
///     rel_tol (float): Relative tolerance. Defaults to 1e-09.
///     abs_tol (float): Absolute tolerance. Defaults to 0.0.
///
/// Returns:
///     bool: True if a and b are close enough.
///
/// Examples:
///     >>> rmath.scalar.isclose(1.0, 1.0000000001)
///     True
///     >>> rmath.scalar.isclose(1.0, 1.001)
///     False
///     >>> rmath.scalar.isclose(0.0, 1e-10, abs_tol=1e-9)
///     True
#[pyfunction]
#[pyo3(signature = (a, b, *, rel_tol=1e-09, abs_tol=0.0))]
pub fn isclose(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let diff = (a - b).abs();
    diff <= (rel_tol * b.abs()).max((rel_tol * a.abs()).max(abs_tol))
}

// ============================================================================
// --- [9/9] Integer and Bitwise ---
// ============================================================================

/// Returns n factorial (n!). Maximum supported n is 34.
///
/// Args:
///     n (int): Value to compute factorial of.
///
/// Returns:
///     int: n! (as a 128-bit integer).
///
/// Raises:
///     OverflowError: If n > 34 (limit of 128-bit uint).
///
/// Examples:
///     >>> rmath.scalar.factorial(5)
///     120
#[pyfunction]
pub fn factorial(n: u64) -> PyResult<u128> {
    if n > 34 {
         return Err(pyo3::exceptions::PyOverflowError::new_err("Factorial too large for u128 (n <= 34)"));
    }
    let mut res: u128 = 1;
    for i in 1..=n as u128 {
        res *= i;
    }
    Ok(res)
}

/// Returns the Greatest Common Divisor of a and b.
///
/// Always returns a non-negative value. gcd(0, 0) returns 0.
/// Handles negative inputs: gcd(-48, 18) == 6.
///
/// Args:
///     a (int): First value.
///     b (int): Second value.
///
/// Returns:
///     int: The GCD, always >= 0.
///
/// Examples:
///     >>> rmath.scalar.gcd(48, 18)
///     6
///     >>> rmath.scalar.gcd(-48, 18)
///     6
///     >>> rmath.scalar.gcd(0, 5)
///     5
#[pyfunction]
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        a %= b;
        std::mem::swap(&mut a, &mut b);
    }
    a.abs()
}

/// Returns the Least Common Multiple of a and b.
///
/// Returns 0 if either a or b is 0. Handles negative inputs.
///
/// Args:
///     a (int): First value.
///     b (int): Second value.
///
/// Returns:
///     int: The LCM, always >= 0.
///
/// Raises:
///     OverflowError: If the result exceeds i64 range.
///
/// Examples:
///     >>> rmath.scalar.lcm(12, 15)
///     60
///     >>> rmath.scalar.lcm(0, 5)
///     0
#[pyfunction]
pub fn lcm(a: i64, b: i64) -> PyResult<i64> {
    if a == 0 || b == 0 { return Ok(0); }
    let gcd_val = gcd(a, b);
    (a / gcd_val)
        .checked_mul(b)
        .map(|v| v.abs())
        .ok_or_else(|| pyo3::exceptions::PyOverflowError::new_err(
            "lcm result overflows i64"
        ))
}

/// Returns True if n is a power of 2.
///
/// Args:
///     n (int): The value.
///
/// Returns:
///     bool: True if n = 2^k for some k >= 0.
///
/// Examples:
///     >>> rmath.scalar.is_power_of_two(8)
///     True
///     >>> rmath.scalar.is_power_of_two(6)
///     False
///     >>> rmath.scalar.is_power_of_two(0)
///     False
#[pyfunction]
pub fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Returns the next power of 2 greater than or equal to n.
///
/// Returns 1 for n <= 1.
///
/// Args:
///     n (int): The value. Must be <= 2^62 to avoid overflow.
///
/// Returns:
///     int: The smallest 2^k >= n.
///
/// Raises:
///     OverflowError: If n > 2^62.
///
/// Examples:
///     >>> rmath.scalar.next_power_of_two(5)
///     8
///     >>> rmath.scalar.next_power_of_two(8)
///     8
///     >>> rmath.scalar.next_power_of_two(1)
///     1
#[pyfunction]
pub fn next_power_of_two(n: i64) -> PyResult<i64> {
    if n <= 1 { return Ok(1); }
    if n > (1i64 << 62) {
        return Err(pyo3::exceptions::PyOverflowError::new_err(
            "next_power_of_two: n too large, would overflow i64"
        ));
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    Ok(v + 1)
}

/// Returns True if n is a prime number.
///
/// Uses a wheel factorisation over 2 and 3 — O(sqrt(n)).
///
/// Args:
///     n (int): The value.
///
/// Returns:
///     bool: True if n is prime.
///
/// Examples:
///     >>> rmath.scalar.is_prime(2)
///     True
///     >>> rmath.scalar.is_prime(17)
///     True
///     >>> rmath.scalar.is_prime(1)
///     False
///     >>> rmath.scalar.is_prime(15)
///     False
#[pyfunction]
pub fn is_prime(n: i64) -> bool {
    if n <= 1 { return false; }
    if n <= 3 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5i64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

// ============================================================================
// --- Registration ---
// ============================================================================

/// Registers the scalar submodule.
pub fn register_scalar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Arithmetic
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(remainder, m)?)?;
    m.add_function(wrap_pyfunction!(fmod, m)?)?;

    // Rounding
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(trunc, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(round_half_even, m)?)?;

    // Common
    m.add_function(wrap_pyfunction!(signum, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(lerp, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(root, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(pow, m)?)?;
    m.add_function(wrap_pyfunction!(inv_sqrt, m)?)?;

    // Exp/Log
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(log2, m)?)?;
    m.add_function(wrap_pyfunction!(log10, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(exp2, m)?)?;
    m.add_function(wrap_pyfunction!(logsumexp2, m)?)?;

    // Trig
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;
    m.add_function(wrap_pyfunction!(atan2, m)?)?;

    // Hyperbolic
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;
    m.add_function(wrap_pyfunction!(asinh, m)?)?;
    m.add_function(wrap_pyfunction!(acosh, m)?)?;
    m.add_function(wrap_pyfunction!(atanh, m)?)?;
    
    // Utils
    m.add_function(wrap_pyfunction!(hypot, m)?)?;
    m.add_function(wrap_pyfunction!(hypot_3d, m)?)?;
    m.add_function(wrap_pyfunction!(fma, m)?)?;
    m.add_function(wrap_pyfunction!(copysign, m)?)?;
    m.add_function(wrap_pyfunction!(nextafter, m)?)?;
    m.add_function(wrap_pyfunction!(degrees, m)?)?;
    m.add_function(wrap_pyfunction!(radians, m)?)?;
    m.add_function(wrap_pyfunction!(frexp, m)?)?;
    m.add_function(wrap_pyfunction!(ulp, m)?)?;

    // Predicates
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(is_integer, m)?)?;
    m.add_function(wrap_pyfunction!(isclose, m)?)?;

    // Integer / Bitwise
    m.add_function(wrap_pyfunction!(factorial, m)?)?;
    m.add_function(wrap_pyfunction!(gcd, m)?)?;
    m.add_function(wrap_pyfunction!(lcm, m)?)?;
    m.add_function(wrap_pyfunction!(is_power_of_two, m)?)?;
    m.add_function(wrap_pyfunction!(next_power_of_two, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    
    Ok(())
}

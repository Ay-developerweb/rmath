use pyo3::prelude::*;
use crate::scalar::core::Scalar;

// ---------------------------------------------------------------------------
// Mathematical methods for Scalar.
//
// NaN policy for *named math methods* (not operators):
//   These methods are called explicitly by the programmer and represent a
//   deliberate mathematical operation.  If the result is NaN due to a domain
//   error (e.g. sqrt of negative, log of zero), we raise PyValueError so the
//   error surfaces clearly.
//
//   This is INTENTIONALLY different from __pow__ in ops.rs, which silently
//   returns NaN because it is an operator and must follow IEEE-754 semantics
//   for interoperability with Python's numeric tower.
//
//   Summary:
//     Scalar(-1.0) ** 0.5   → Scalar(NaN)          [operator, no raise]
//     Scalar(-1.0).sqrt()   → PyValueError          [method, strict]
// ---------------------------------------------------------------------------

#[pymethods]
impl Scalar {
    // -----------------------------------------------------------------------
    // Root / power
    // -----------------------------------------------------------------------

    /// Square root.
    ///
    /// Raises `ValueError` if the input is negative.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(16.0).sqrt()
    ///     Scalar(4.0)
    pub fn sqrt(&self) -> PyResult<Scalar> {
        if self.0 < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sqrt: domain error — cannot take square root of a negative Scalar",
            ));
        }
        Ok(Scalar(self.0.sqrt()))
    }

    /// Cube root.
    ///
    /// Defined for all real numbers (including negatives).
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(27.0).cbrt()
    ///     Scalar(3.0)
    ///     >>> Scalar(-8.0).cbrt()
    ///     Scalar(-2.0)
    pub fn cbrt(&self) -> Scalar {
        Scalar(self.0.cbrt())
    }

    /// Raise to a real power.
    ///
    /// Raises `ValueError` if the result is NaN (e.g. negative base with fractional exponent).
    /// This is a stricter version of the `**` operator.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.0).pow(3.0)
    ///     Scalar(8.0)
    pub fn pow(&self, exp: f64) -> PyResult<Scalar> {
        let result = self.0.powf(exp);
        if result.is_nan() && !self.0.is_nan() {
            // NaN came from a domain error, not from NaN propagation.
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pow: math domain error — Scalar({}).pow({})",
                self.0, exp
            )));
        }
        Ok(Scalar(result))
    }

    // -----------------------------------------------------------------------
    // Exponential and logarithm
    // -----------------------------------------------------------------------

    /// Natural exponential: e^x.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(1.0).exp()
    ///     Scalar(2.718281828459045)
    pub fn exp(&self) -> Scalar {
        Scalar(self.0.exp())
    }

    /// Base-2 exponential: 2^x.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(10.0).exp2()
    ///     Scalar(1024.0)
    pub fn exp2(&self) -> Scalar {
        Scalar(self.0.exp2())
    }

    /// Natural logarithm (base e).
    ///
    /// Raises `ValueError` for non-positive input.
    /// Pass `base` to compute log in an arbitrary base.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar, e
    ///     >>> Scalar(e).log()
    ///     Scalar(1.0)
    ///     >>> Scalar(100.0).log(base=10.0)
    ///     Scalar(2.0)
    #[pyo3(signature = (base = None))]
    pub fn log(&self, base: Option<f64>) -> PyResult<Scalar> {
        if self.0 <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "log: domain error — input must be positive",
            ));
        }
        match base {
            None => Ok(Scalar(self.0.ln())),
            Some(b) => {
                if b <= 0.0 || b == 1.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "log: base must be positive and not equal to 1",
                    ));
                }
                Ok(Scalar(self.0.log(b)))
            }
        }
    }

    /// Base-2 logarithm.
    ///
    /// Raises `ValueError` for non-positive input.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(1024.0).log2()
    ///     Scalar(10.0)
    pub fn log2(&self) -> PyResult<Scalar> {
        if self.0 <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "log2: domain error — input must be positive",
            ));
        }
        Ok(Scalar(self.0.log2()))
    }

    /// Base-10 logarithm.
    ///
    /// Raises `ValueError` for non-positive input.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(1000.0).log10()
    ///     Scalar(3.0)
    pub fn log10(&self) -> PyResult<Scalar> {
        if self.0 <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "log10: domain error — input must be positive",
            ));
        }
        Ok(Scalar(self.0.log10()))
    }

    // -----------------------------------------------------------------------
    // Trigonometric (input in radians)
    // -----------------------------------------------------------------------

    /// Sine (input in radians).
    ///
    /// Examples:
    ///     >>> from rmath import Scalar, pi
    ///     >>> Scalar(pi/2).sin()
    ///     Scalar(1.0)
    pub fn sin(&self) -> Scalar {
        Scalar(self.0.sin())
    }

    /// Cosine (input in radians).
    ///
    /// Examples:
    ///     >>> from rmath import Scalar, pi
    ///     >>> Scalar(pi).cos()
    ///     Scalar(-1.0)
    pub fn cos(&self) -> Scalar {
        Scalar(self.0.cos())
    }

    /// Tangent (input in radians).
    ///
    /// Examples:
    ///     >>> from rmath import Scalar, pi
    ///     >>> Scalar(pi/4).tan()
    ///     Scalar(1.0)
    pub fn tan(&self) -> Scalar {
        Scalar(self.0.tan())
    }

    // -----------------------------------------------------------------------
    // Inverse trigonometric (return radians)
    // -----------------------------------------------------------------------

    /// Arcsine (returns radians).
    ///
    /// Raises `ValueError` if input is outside `[-1, 1]`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(1.0).asin()
    ///     Scalar(1.5707963267948966)
    pub fn asin(&self) -> PyResult<Scalar> {
        if self.0 < -1.0 || self.0 > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "asin: domain error — input must be in [-1, 1]",
            ));
        }
        Ok(Scalar(self.0.asin()))
    }

    /// Arccosine (returns radians).
    ///
    /// Raises `ValueError` if input is outside `[-1, 1]`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(-1.0).acos()
    ///     Scalar(3.141592653589793)
    pub fn acos(&self) -> PyResult<Scalar> {
        if self.0 < -1.0 || self.0 > 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "acos: domain error — input must be in [-1, 1]",
            ));
        }
        Ok(Scalar(self.0.acos()))
    }

    /// Arctangent.
    ///
    /// Returns a value in `(-π/2, π/2)`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(1.0).atan()
    ///     Scalar(0.7853981633974483)
    pub fn atan(&self) -> Scalar {
        Scalar(self.0.atan())
    }

    /// Two-argument arctangent of `(self, x)`.
    ///
    /// Returns the angle in `(-π, π]` — equivalent to `atan2(y, x)` where `y=self`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(1.0).atan2(1.0)
    ///     Scalar(0.7853981633974483)
    pub fn atan2(&self, x: f64) -> Scalar {
        Scalar(self.0.atan2(x))
    }

    // -----------------------------------------------------------------------
    // Hyperbolic
    // -----------------------------------------------------------------------

    /// Hyperbolic sine.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(0.0).sinh()
    ///     Scalar(0.0)
    pub fn sinh(&self) -> Scalar {
        Scalar(self.0.sinh())
    }

    /// Hyperbolic cosine.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(0.0).cosh()
    ///     Scalar(1.0)
    pub fn cosh(&self) -> Scalar {
        Scalar(self.0.cosh())
    }

    /// Hyperbolic tangent.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(0.0).tanh()
    ///     Scalar(0.0)
    pub fn tanh(&self) -> Scalar {
        Scalar(self.0.tanh())
    }

    /// Inverse hyperbolic sine.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(0.0).asinh()
    ///     Scalar(0.0)
    pub fn asinh(&self) -> Scalar {
        Scalar(self.0.asinh())
    }

    /// Inverse hyperbolic cosine.
    ///
    /// Raises `ValueError` if input is less than 1.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(1.0).acosh()
    ///     Scalar(0.0)
    pub fn acosh(&self) -> PyResult<Scalar> {
        if self.0 < 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "acosh: domain error — input must be >= 1",
            ));
        }
        Ok(Scalar(self.0.acosh()))
    }

    /// Inverse hyperbolic tangent.
    ///
    /// Raises `ValueError` if input is outside `(-1, 1)`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(0.0).atanh()
    ///     Scalar(0.0)
    pub fn atanh(&self) -> PyResult<Scalar> {
        if self.0 <= -1.0 || self.0 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "atanh: domain error — input must be in (-1, 1)",
            ));
        }
        Ok(Scalar(self.0.atanh()))
    }

    // -----------------------------------------------------------------------
    // Rounding
    // -----------------------------------------------------------------------

    /// Round toward positive infinity.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.1).ceil()
    ///     Scalar(3.0)
    pub fn ceil(&self) -> Scalar {
        Scalar(self.0.ceil())
    }

    /// Round toward negative infinity.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.9).floor()
    ///     Scalar(2.0)
    pub fn floor(&self) -> Scalar {
        Scalar(self.0.floor())
    }

    /// Round to the nearest integer, ties away from zero.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.5).round()
    ///     Scalar(3.0)
    ///     >>> Scalar(-2.5).round()
    ///     Scalar(-3.0)
    pub fn round(&self) -> Scalar {
        Scalar(self.0.round())
    }

    /// Truncate toward zero (drop the fractional part).
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.9).trunc()
    ///     Scalar(2.0)
    ///     >>> Scalar(-2.9).trunc()
    ///     Scalar(-2.0)
    pub fn trunc(&self) -> Scalar {
        Scalar(self.0.trunc())
    }

    /// Fractional part: `self - self.trunc()`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.75).fract()
    ///     Scalar(0.75)
    pub fn fract(&self) -> Scalar {
        Scalar(self.0.fract())
    }

    // -----------------------------------------------------------------------
    // Absolute value (also in ops.rs as __abs__ for operator support;
    // this .abs() method form is the explicit call version)
    // -----------------------------------------------------------------------

    /// Absolute value.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(-5.0).abs()
    ///     Scalar(5.0)
    pub fn abs(&self) -> Scalar {
        Scalar(self.0.abs())
    }

    // -----------------------------------------------------------------------
    // Hypotenuse (Pythagorean addition, numerically stable)
    // -----------------------------------------------------------------------

    /// Numerically stable `sqrt(self² + other²)`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(3.0).hypot(4.0)
    ///     Scalar(5.0)
    pub fn hypot(&self, other: f64) -> Scalar {
        Scalar(self.0.hypot(other))
    }
}
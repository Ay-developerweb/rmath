use pyo3::prelude::*;
use crate::scalar::core::{Scalar, extract_f64};

// ---------------------------------------------------------------------------
// All arithmetic dunder methods for Scalar.
//
// Design rules enforced here:
//   1. Every binary op accepts Scalar | float | int on the RHS via extract_f64.
//   2. Reflected ops (__radd__ etc.) also accept Scalar, not just f64.
//   3. __pow__ does NOT raise on NaN — NaN is a valid IEEE-754 result.
//      The separate `.pow()` math method (in math.rs) may choose to raise.
//   4. In-place ops (__iadd__ etc.) return a new Scalar — Scalar is Copy
//      so mutation-in-place is semantically identical to a new value.
// ---------------------------------------------------------------------------

#[pymethods]
impl Scalar {
    // -----------------------------------------------------------------------
    // Addition
    // -----------------------------------------------------------------------

    /// Addition operator: self + other.
    ///
    /// Supports addition with other Scalar instances, ints, or floats.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(10.0) + 5.0
    ///     Scalar(15.0)
    ///     >>> Scalar(10.0) + Scalar(2.0)
    ///     Scalar(12.0)
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        Ok(Scalar(self.0 + extract_f64(other)?))
    }

    /// Reflected addition: other + self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> 5.0 + Scalar(10.0)
    ///     Scalar(15.0)
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        // other + self
        Ok(Scalar(extract_f64(other)? + self.0))
    }

    // -----------------------------------------------------------------------
    // Subtraction
    // -----------------------------------------------------------------------

    /// Subtraction operator: self - other.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(10.0) - 4.5
    ///     Scalar(5.5)
    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        Ok(Scalar(self.0 - extract_f64(other)?))
    }

    /// Reflected subtraction: other - self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> 10.0 - Scalar(4.0)
    ///     Scalar(6.0)
    pub fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        // other - self
        Ok(Scalar(extract_f64(other)? - self.0))
    }

    // -----------------------------------------------------------------------
    // Multiplication
    // -----------------------------------------------------------------------

    /// Multiplication operator: self * other.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.5) * 4
    ///     Scalar(10.0)
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        Ok(Scalar(self.0 * extract_f64(other)?))
    }

    /// Reflected multiplication: other * self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> 2.0 * Scalar(3.5)
    ///     Scalar(7.0)
    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        // other * self  (commutative, but we still need the reflected form)
        Ok(Scalar(extract_f64(other)? * self.0))
    }

    // -----------------------------------------------------------------------
    // True division
    // -----------------------------------------------------------------------

    /// True division operator: self / other.
    ///
    /// Raises `ZeroDivisionError` if other is 0.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(10.0) / 4
    ///     Scalar(2.5)
    pub fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        let rhs = extract_f64(other)?;
        if rhs == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Scalar division by zero",
            ));
        }
        Ok(Scalar(self.0 / rhs))
    }

    /// Reflected true division: other / self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> 1.0 / Scalar(4.0)
    ///     Scalar(0.25)
    pub fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        // other / self
        if self.0 == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Scalar division by zero",
            ));
        }
        Ok(Scalar(extract_f64(other)? / self.0))
    }

    // -----------------------------------------------------------------------
    // Floor division
    // -----------------------------------------------------------------------

    /// Floor division operator: self // other.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(10.0) // 3
    ///     Scalar(3.0)
    pub fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        let rhs = extract_f64(other)?;
        if rhs == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Scalar floor division by zero",
            ));
        }
        Ok(Scalar((self.0 / rhs).floor()))
    }

    /// Reflected floor division: other // self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> 10.0 // Scalar(3.0)
    ///     Scalar(3.0)
    pub fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        if self.0 == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Scalar floor division by zero",
            ));
        }
        Ok(Scalar((extract_f64(other)? / self.0).floor()))
    }

    // -----------------------------------------------------------------------
    // Modulo
    // -----------------------------------------------------------------------

    /// Modulo operator: self % other.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(10.0) % 3
    ///     Scalar(1.0)
    pub fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        let rhs = extract_f64(other)?;
        if rhs == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Scalar modulo by zero",
            ));
        }
        Ok(Scalar(self.0 % rhs))
    }

    /// Reflected modulo: other % self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> 10.0 % Scalar(3.0)
    ///     Scalar(1.0)
    pub fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        if self.0 == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Scalar modulo by zero",
            ));
        }
        Ok(Scalar(extract_f64(other)? % self.0))
    }

    // -----------------------------------------------------------------------
    // Power
    //
    // __pow__ intentionally does NOT raise on NaN results.
    // IEEE-754 defines e.g. (-1.0).powf(0.5) = NaN — that is the correct
    // mathematical answer in the real domain.  If the programmer wants an
    // error on NaN, they should use the `.pow()` method from math.rs which
    // is documented as strict.
    //
    // The `modulo` argument exists for Python's three-argument pow(x, y, mod).
    // We do not support it for floats (Python itself raises TypeError for it)
    // so we raise a clear error if it is provided.
    // -----------------------------------------------------------------------

    /// Power operator: self ** other.
    ///
    /// Matches IEEE-754 semantics: returns NaN for domain errors (e.g. -1 ** 0.5)
    /// rather than raising. For strict error checking, use `.pow()`.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> Scalar(2.0) ** 3
    ///     Scalar(8.0)
    ///     >>> Scalar(-1.0) ** 0.5
    ///     Scalar(nan)
    pub fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Scalar> {
        if modulo.is_some() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Scalar does not support three-argument pow() with a modulus",
            ));
        }
        Ok(Scalar(self.0.powf(extract_f64(other)?)))
    }

    /// Reflected power operator: other ** self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> 2.0 ** Scalar(3.0)
    ///     Scalar(8.0)
    pub fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Scalar> {
        // other ** self
        if modulo.is_some() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Scalar does not support three-argument pow() with a modulus",
            ));
        }
        Ok(Scalar(extract_f64(other)?.powf(self.0)))
    }

    // -----------------------------------------------------------------------
    // Unary operators
    // -----------------------------------------------------------------------

    /// Unary negation: -self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> -Scalar(42.0)
    ///     Scalar(-42.0)
    pub fn __neg__(&self) -> Scalar {
        Scalar(-self.0)
    }

    /// Unary positive: +self.
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> +Scalar(42.0)
    ///     Scalar(42.0)
    pub fn __pos__(&self) -> Scalar {
        *self
    }

    /// Absolute value: abs(self).
    ///
    /// Examples:
    ///     >>> from rmath import Scalar
    ///     >>> abs(Scalar(-5.0))
    ///     Scalar(5.0)
    pub fn __abs__(&self) -> Scalar {
        Scalar(self.0.abs())
    }
}
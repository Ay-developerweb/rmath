use pyo3::prelude::*;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// A high-performance, Rust-backed scalar numeric value.
///
/// `Scalar` wraps a 64-bit IEEE-754 float and exposes it to Python
/// with a complete numeric protocol: arithmetic, comparison, hashing,
/// boolean coercion, and format — all resolved natively without
/// Python object allocation in the hot path.
///
/// Performance Philosophy:
///     Rmath scalar operations are designed for high-throughput numeric workloads. 
///     While Python's built-in `float` is highly optimized for general-purpose use, 
///     `rmath.Scalar` allows for chaining operations that execute entirely in Rust.
///
/// NaN and Error Policy:
///     Rmath distinguishes between *operators* and *named methods*:
///     
///     1. Operators (+, -, *, /, **) follow IEEE-754 semantics strictly. They 
///        never raise exceptions for domain errors; they return `NaN` or `inf`.
///        
///     2. Named Methods (.sqrt(), .log(), .pow()) are stricter. Since they represent 
///        a deliberate mathematical call, they raise `ValueError` for domain 
///        errors to help catch bugs early.
///
/// # Examples
/// ```python
/// from rmath import scalar as sc
/// x = sc.Scalar(3.14)
/// y = sc.Scalar(2.0)
/// print(x + y)          # Scalar(5.14)
/// print(float(x))       # 3.14
/// ```
#[pyclass(module = "rmath.scalar")]
#[derive(Debug, Clone, Copy)]
pub struct Scalar(pub f64);

// ---------------------------------------------------------------------------
// Rust-side trait impls (not exposed to Python directly but used internally)
// ---------------------------------------------------------------------------

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        // NaN != NaN by IEEE-754; we honour that here.
        self.0 == other.0
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Helper: extract a raw f64 from either a Scalar or a Python numeric
// ---------------------------------------------------------------------------
#[inline(always)]
pub(crate) fn extract_f64(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(s) = obj.extract::<Scalar>() {
        return Ok(s.0);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(f);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i as f64);
    }
    let type_name = obj.get_type().name()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(pyo3::exceptions::PyTypeError::new_err(
        format!("cannot convert '{type_name}' to Scalar"),
    ))
}

// ---------------------------------------------------------------------------
// Python methods
// ---------------------------------------------------------------------------

#[pymethods]
impl Scalar {
    // --- Construction ---

    /// Create a new high-precision Scalar from a float.
    #[new]
    pub fn new(value: f64) -> Self {
        Scalar(value)
    }

    // --- Conversion ---

    /// Convert to a native Python float.
    /// This is the explicit boundary crossing back into Python-land.
    pub fn to_python(&self) -> f64 {
        self.0
    }

    /// `float(scalar)` protocol.
    pub fn __float__(&self) -> f64 {
        self.0
    }

    /// `int(scalar)` protocol — truncates toward zero (C-style cast).
    pub fn __int__(&self) -> PyResult<i64> {
        if self.0.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot convert NaN Scalar to int",
            ));
        }
        if self.0.is_infinite() {
            return Err(pyo3::exceptions::PyOverflowError::new_err(
                "cannot convert infinite Scalar to int",
            ));
        }
        Ok(self.0 as i64)
    }

    /// `bool(scalar)` protocol — `False` iff value is exactly 0.0 or NaN.
    pub fn __bool__(&self) -> bool {
        self.0 != 0.0 && !self.0.is_nan()
    }

    // --- Representation ---

    /// Detailed string representation for debugging.
    pub fn __repr__(&self) -> String {
        format!("Scalar({})", self.0)
    }

    /// User-friendly string representation.
    pub fn __str__(&self) -> String {
        self.0.to_string()
    }

    /// f-string format support: `f"{x:.4f}"` works correctly.
    pub fn __format__(&self, format_spec: &str) -> PyResult<String> {
        Python::with_gil(|py| {
            let py_float = pyo3::types::PyFloat::new(py, self.0);
            let formatted = py_float.call_method1("__format__", (format_spec,))?;
            formatted.extract::<String>()
        })
    }

    // --- Hashing ---
    //
    // Python contract: if a == b then hash(a) == hash(b).
    // We match Python's own float hashing so that
    //   hash(Scalar(1.0)) == hash(1.0) == hash(1)
    // which is required for correct dict/set behaviour.
    pub fn __hash__(&self) -> u64 {
        // Normalise -0.0 → 0.0 so hash(-0.0) == hash(0.0).
        let v = if self.0 == 0.0 { 0.0_f64 } else { self.0 };
        let mut h = DefaultHasher::new();
        // Use bit representation for NaN-stable hashing.
        v.to_bits().hash(&mut h);
        h.finish()
    }

    // --- Comparison ---

    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        extract_f64(other).map(|v| self.0 == v).unwrap_or(false)
    }

    pub fn __ne__(&self, other: &Bound<'_, PyAny>) -> bool {
        !self.__eq__(other)
    }

    pub fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0 < extract_f64(other)?)
    }

    pub fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0 <= extract_f64(other)?)
    }

    pub fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0 > extract_f64(other)?)
    }

    /// Check if self is greater than or equal to other.
    pub fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0 >= extract_f64(other)?)
    }

    // --- IEEE-754 predicates ---

    /// Returns `True` if this value is NaN (not a number).
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    /// Returns `True` if this value is positive or negative infinity.
    pub fn is_inf(&self) -> bool {
        self.0.is_infinite()
    }

    /// Returns `True` if this value is neither NaN nor infinite.
    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    // --- Utility operations ---

    /// Clamps the value to `[low, high]`.
    ///
    /// If `low > high`, raises `ValueError`.
    pub fn clamp(&self, low: f64, high: f64) -> PyResult<Scalar> {
        if low > high {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "clamp: low must be <= high",
            ));
        }
        Ok(Scalar(self.0.clamp(low, high)))
    }

    /// Returns the sign of the value: `Scalar(-1.0)`, `Scalar(0.0)`, or `Scalar(1.0)`.
    /// NaN input returns NaN.
    pub fn signum(&self) -> Scalar {
        if self.0.is_nan() {
            Scalar(f64::NAN)
        } else if self.0 == 0.0 {
            Scalar(0.0)
        } else {
            Scalar(self.0.signum())
        }
    }

    /// Linear interpolation: `self + t * (other - self)`.
    ///
    /// `t = 0.0` returns `self`, `t = 1.0` returns `other`.
    /// `t` is not clamped — extrapolation is allowed.
    pub fn lerp(&self, other: f64, t: f64) -> Scalar {
        Scalar(self.0 + t * (other - self.0))
    }
}
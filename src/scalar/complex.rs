use pyo3::prelude::*;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use crate::scalar::core::Scalar;

// ---------------------------------------------------------------------------
// Complex number: re + im·i, backed by two f64 values (16 bytes on stack).
//
// Distinct from Scalar by design — users working purely in ℝ never pay the
// branch-prediction or cognitive cost of complex arithmetic paths.
//
// Implements the full Python numeric protocol:
//   - All arithmetic operators and their reflected forms
//   - __abs__, __neg__, __pos__, __bool__
//   - __hash__, __eq__, __ne__
//   - __complex__ for Python builtin interop
//   - Mathematical methods: log, sqrt, pow, sin, cos
//   - Polar coordinate form: to_polar() / from_polar()
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helper: extract a Complex from either a Complex object or a real number.
// ---------------------------------------------------------------------------
#[inline(always)]
fn extract_complex(obj: &Bound<'_, PyAny>) -> PyResult<Complex> {
    if let Ok(c) = obj.extract::<Complex>() {
        return Ok(c);
    }
    // A real number is a complex number with imaginary part zero.
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Complex { re: f, im: 0.0 });
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Complex { re: i as f64, im: 0.0 });
    }
    if let Ok(s) = obj.extract::<Scalar>() {
        return Ok(Complex { re: s.0, im: 0.0 });
    }
    let type_name = obj.get_type().name()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "cannot convert '{type_name}' to Complex",
    )))
}

#[pyclass(module = "rmath")]
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    #[pyo3(get, set)]
    pub re: f64,
    #[pyo3(get, set)]
    pub im: f64,
}

#[pymethods]
impl Complex {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new complex number: `re + im*j`.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(3.0, 4.0)
    ///     (3.000000+4.000000j)
    #[new]
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    /// Construct from polar coordinates (r, θ).
    ///
    /// `r` is the modulus (must be >= 0), `theta` is the argument in radians.
    #[staticmethod]
    pub fn from_polar(r: f64, theta: f64) -> PyResult<Complex> {
        if r < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "from_polar: modulus r must be >= 0",
            ));
        }
        Ok(Complex {
            re: r * theta.cos(),
            im: r * theta.sin(),
        })
    }

    // -----------------------------------------------------------------------
    // Properties / projection
    // -----------------------------------------------------------------------

    /// Modulus |z| = √(re² + im²).
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(3.0, 4.0).abs()
    ///     5.0
    pub fn abs(&self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Argument (phase angle) of z in radians, in `(-π, π]`.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(0.0, 1.0).arg()
    ///     1.5707963267948966
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate: `re - im·i`.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(3.0, 4.0).conjugate()
    ///     (3.000000-4.000000j)
    pub fn conjugate(&self) -> Self {
        Complex { re: self.re, im: -self.im }
    }

    /// Convert to polar form: returns `(modulus, argument)`.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(0.0, 2.0).to_polar()
    ///     (2.0, 1.5707963267948966)
    pub fn to_polar(&self) -> (f64, f64) {
        (self.abs(), self.arg())
    }

    // -----------------------------------------------------------------------
    // Mathematical functions
    // -----------------------------------------------------------------------

    /// Complex exponential: e^z.
    ///
    /// Examples:
    ///     >>> from rmath import Complex, pi
    ///     >>> Complex(0.0, pi).exp()
    ///     (-1.000000+0.000000j)
    pub fn exp(&self) -> Self {
        let exp_re = self.re.exp();
        Complex {
            re: exp_re * self.im.cos(),
            im: exp_re * self.im.sin(),
        }
    }

    /// Complex natural logarithm: ln(z).
    ///
    /// Raises `ValueError` if z is zero.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(1.0, 1.0).log()
    ///     (0.346574+0.785398j)
    pub fn log(&self) -> PyResult<Complex> {
        let modulus = self.abs();
        if modulus == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "log: complex logarithm of zero is undefined",
            ));
        }
        Ok(Complex {
            re: modulus.ln(),
            im: self.arg(),
        })
    }

    /// Complex square root (principal branch).
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(-4.0, 0.0).sqrt()
    ///     (0.000000+2.000000j)
    pub fn sqrt(&self) -> Complex {
        let modulus = self.abs();
        // Principal branch formula:
        //   w = sqrt((|z| + re) / 2)
        //   if im >= 0: result = (w, im / (2w))
        //   else:       result = (|im| / (2w), -w)
        let w = ((modulus + self.re) / 2.0).sqrt();
        if w == 0.0 {
            Complex { re: 0.0, im: 0.0 }
        } else if self.im >= 0.0 {
            Complex { re: w, im: self.im / (2.0 * w) }
        } else {
            Complex { re: (-self.im) / (2.0 * w), im: -w }
        }
    }

    /// Complex power: z^other.
    ///
    /// Raises `ValueError` if base is zero and exponent has non-positive real part.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(0.0, 1.0) ** 2
    ///     (-1.000000+0.000000j)
    pub fn pow(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        let exp = extract_complex(other)?;
        // 0^w is 0 when Re(w) > 0, undefined otherwise.
        if self.re == 0.0 && self.im == 0.0 {
            if exp.re > 0.0 {
                return Ok(Complex { re: 0.0, im: 0.0 });
            }
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pow: 0^w is undefined for Re(w) <= 0",
            ));
        }
        // z^w = exp(w * ln(z))
        let ln_z = self.log()?;
        let product = Complex {
            re: exp.re * ln_z.re - exp.im * ln_z.im,
            im: exp.re * ln_z.im + exp.im * ln_z.re,
        };
        Ok(product.exp())
    }

    /// Complex sine.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(1.0, 1.0).sin()
    ///     (1.298458+0.634964j)
    pub fn sin(&self) -> Complex {
        Complex {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// Complex cosine.
    ///
    /// Examples:
    ///     >>> from rmath import Complex
    ///     >>> Complex(1.0, 1.0).cos()
    ///     (0.833730-0.988898j)
    pub fn cos(&self) -> Complex {
        Complex {
            re: self.re.cos() * self.im.cosh(),
            im: -(self.re.sin() * self.im.sinh()),
        }
    }

    // -----------------------------------------------------------------------
    // Python numeric protocol
    // -----------------------------------------------------------------------

    /// Absolute value (modulus).
    pub fn __abs__(&self) -> f64 {
        self.abs()
    }

    /// Unary negation: -z.
    pub fn __neg__(&self) -> Complex {
        Complex { re: -self.re, im: -self.im }
    }

    /// Unary positive: +z.
    pub fn __pos__(&self) -> Complex {
        *self
    }

    /// Boolean coercion: `True` if z != 0.
    pub fn __bool__(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }

    /// `complex(z)` — interoperability with Python's built-in complex type.
    pub fn __complex__(&self) -> (f64, f64) {
        (self.re, self.im)
    }

    // --- Equality and hashing ---

    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        extract_complex(other)
            .map(|c| self.re == c.re && self.im == c.im)
            .unwrap_or(false)
    }

    pub fn __ne__(&self, other: &Bound<'_, PyAny>) -> bool {
        !self.__eq__(other)
    }

    /// Hash based on the component pair.
    /// Satisfies: `hash(Complex(x, 0)) == hash(Scalar(x))` is NOT guaranteed
    /// (they are different types), but `Complex(a,b) == Complex(a,b)` implies
    /// equal hashes, which is the Python contract.
    pub fn __hash__(&self) -> u64 {
        let re = if self.re == 0.0 { 0.0_f64 } else { self.re };
        let im = if self.im == 0.0 { 0.0_f64 } else { self.im };
        let mut h = DefaultHasher::new();
        re.to_bits().hash(&mut h);
        im.to_bits().hash(&mut h);
        h.finish()
    }

    // --- Ordering: complex numbers are not fully ordered, but we support
    //     comparison by modulus so they can be used in min/max contexts. ---

    pub fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.abs() < extract_complex(other)?.abs())
    }

    pub fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.abs() <= extract_complex(other)?.abs())
    }

    pub fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.abs() > extract_complex(other)?.abs())
    }

    pub fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.abs() >= extract_complex(other)?.abs())
    }

    // -----------------------------------------------------------------------
    // Arithmetic operators
    // -----------------------------------------------------------------------

    /// Addition operator: self + other.
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        let c = extract_complex(other)?;
        Ok(Complex { re: self.re + c.re, im: self.im + c.im })
    }

    /// Reflected addition: other + self.
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        // addition is commutative
        self.__add__(other)
    }

    /// Subtraction operator: self - other.
    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        let c = extract_complex(other)?;
        Ok(Complex { re: self.re - c.re, im: self.im - c.im })
    }

    /// Reflected subtraction: other - self.
    pub fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        // other - self
        let c = extract_complex(other)?;
        Ok(Complex { re: c.re - self.re, im: c.im - self.im })
    }

    /// Multiplication operator: self * other.
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        let c = extract_complex(other)?;
        Ok(Complex {
            re: self.re * c.re - self.im * c.im,
            im: self.re * c.im + self.im * c.re,
        })
    }

    /// Reflected multiplication: other * self.
    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        // multiplication is commutative for complex numbers
        self.__mul__(other)
    }

    /// Division operator: self / other.
    pub fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        let c = extract_complex(other)?;
        let denom = c.re * c.re + c.im * c.im;
        if denom == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Complex division by zero",
            ));
        }
        Ok(Complex {
            re: (self.re * c.re + self.im * c.im) / denom,
            im: (self.im * c.re - self.re * c.im) / denom,
        })
    }

    /// Reflected division: other / self.
    pub fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        // other / self
        let c = extract_complex(other)?;
        let denom = self.re * self.re + self.im * self.im;
        if denom == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Complex division by zero",
            ));
        }
        Ok(Complex {
            re: (c.re * self.re + c.im * self.im) / denom,
            im: (c.im * self.re - c.re * self.im) / denom,
        })
    }

    /// Power operator: self ** other.
    pub fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Complex> {
        if modulo.is_some() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Complex does not support three-argument pow()",
            ));
        }
        self.pow(other)
    }

    /// Reflected power operator: base ** self.
    pub fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Complex> {
        if modulo.is_some() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Complex does not support three-argument pow()",
            ));
        }
        // other ** self  →  base.pow(self)
        let base = extract_complex(other)?;
        let exp = *self;

        // 0^w: defined for Re(w)>0, undefined otherwise
        if base.re == 0.0 && base.im == 0.0 {
            if exp.re > 0.0 {
                return Ok(Complex { re: 0.0, im: 0.0 });
            }
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pow: 0^w is undefined for Re(w) <= 0",
            ));
        }
        // base^exp = exp(exp * ln(base))
        let modulus = base.abs();
        let ln_base = Complex {
            re: modulus.ln(),
            im: base.arg(),
        };
        let product = Complex {
            re: exp.re * ln_base.re - exp.im * ln_base.im,
            im: exp.re * ln_base.im + exp.im * ln_base.re,
        };
        Ok(product.exp())
    }

    // -----------------------------------------------------------------------
    // Representation
    // -----------------------------------------------------------------------

    pub fn __repr__(&self) -> String {
        // Standard mathematical form: (re+imj) or (re-imj)
        if self.im >= 0.0 {
            format!("({:.6}+{:.6}j)", self.re, self.im)
        } else {
            format!("({:.6}{:.6}j)", self.re, self.im)
        }
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}
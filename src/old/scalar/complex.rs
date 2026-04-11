use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    #[pyo3(get, set)]
    pub re: f64,
    #[pyo3(get, set)]
    pub im: f64,
}

#[pymethods]
impl Complex {
    #[new]
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    /// Returns the absolute value (magnitude) of the complex number.
    pub fn abs(&self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Returns the phase (angle) of the complex number in radians.
    pub fn arg(&self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Returns the complex conjugate.
    pub fn conjugate(&self) -> Self {
        Complex { re: self.re, im: -self.im }
    }

    /// Complex Exponential: e^(a + bi) = e^a * (cos(b) + i sin(b))
    pub fn exp(&self) -> Self {
        let exp_re = self.re.exp();
        Complex {
            re: exp_re * self.im.cos(),
            im: exp_re * self.im.sin(),
        }
    }

    pub fn __repr__(&self) -> String {
        if self.im >= 0.0 {
            format!("Complex({} + {}j)", self.re, self.im)
        } else {
            format!("Complex({} - {}j)", self.re, -self.im)
        }
    }

    // --- Arithmetic ---
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        if let Ok(other_c) = other.extract::<Complex>() {
            return Ok(Complex { re: self.re + other_c.re, im: self.im + other_c.im });
        }
        if let Ok(other_f) = other.extract::<f64>() {
            return Ok(Complex { re: self.re + other_f, im: self.im });
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for Complex addition"))
    }

    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        if let Ok(other_c) = other.extract::<Complex>() {
            return Ok(Complex { re: self.re - other_c.re, im: self.im - other_c.im });
        }
        if let Ok(other_f) = other.extract::<f64>() {
            return Ok(Complex { re: self.re - other_f, im: self.im });
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for Complex subtraction"))
    }

    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        if let Ok(other_c) = other.extract::<Complex>() {
            return Ok(Complex {
                re: self.re * other_c.re - self.im * other_c.im,
                im: self.re * other_c.im + self.im * other_c.re,
            });
        }
        if let Ok(other_f) = other.extract::<f64>() {
            return Ok(Complex { re: self.re * other_f, im: self.im * other_f });
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for Complex multiplication"))
    }

    pub fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Complex> {
        if let Ok(other_c) = other.extract::<Complex>() {
            let denom = other_c.re * other_c.re + other_c.im * other_c.im;
            if denom == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err("Complex division by zero"));
            }
            return Ok(Complex {
                re: (self.re * other_c.re + self.im * other_c.im) / denom,
                im: (self.im * other_c.re - self.re * other_c.im) / denom,
            });
        }
        if let Ok(other_f) = other.extract::<f64>() {
            if other_f == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err("Complex division by zero"));
            }
            return Ok(Complex { re: self.re / other_f, im: self.im / other_f });
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for Complex division"))
    }
}

use pyo3::prelude::*;
use crate::scalar::core::Scalar;

#[pymethods]
impl Scalar {
    // --- Addition ---
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        if let Ok(other_sc) = other.extract::<Scalar>() {
            return Ok(Scalar(self.0 + other_sc.0));
        }
        if let Ok(other_f) = other.extract::<f64>() {
            return Ok(Scalar(self.0 + other_f));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Unsupported operand type for +"))
    }

    pub fn __radd__(&self, other: f64) -> Scalar {
        Scalar(other + self.0)
    }

    // --- Subtraction ---
    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        if let Ok(other_sc) = other.extract::<Scalar>() {
            return Ok(Scalar(self.0 - other_sc.0));
        }
        if let Ok(other_f) = other.extract::<f64>() {
            return Ok(Scalar(self.0 - other_f));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Unsupported operand type for -"))
    }

    pub fn __rsub__(&self, other: f64) -> Scalar {
        Scalar(other - self.0)
    }

    // --- Multiplication ---
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        if let Ok(other_sc) = other.extract::<Scalar>() {
            return Ok(Scalar(self.0 * other_sc.0));
        }
        if let Ok(other_f) = other.extract::<f64>() {
            return Ok(Scalar(self.0 * other_f));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Unsupported operand type for *"))
    }

    pub fn __rmul__(&self, other: f64) -> Scalar {
        Scalar(other * self.0)
    }

    // --- Division ---
    pub fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        let div_by = if let Ok(other_sc) = other.extract::<Scalar>() {
            other_sc.0
        } else if let Ok(other_f) = other.extract::<f64>() {
            other_f
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Unsupported operand type for /"));
        };

        if div_by == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err("Scalar division by zero"));
        }
        Ok(Scalar(self.0 / div_by))
    }

    pub fn __rtruediv__(&self, other: f64) -> PyResult<Scalar> {
        if self.0 == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err("Scalar division by zero"));
        }
        Ok(Scalar(other / self.0))
    }

    // --- Powers ---
    pub fn __pow__(&self, other: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Scalar> {
        let exp = if let Ok(other_sc) = other.extract::<Scalar>() {
            other_sc.0
        } else if let Ok(other_f) = other.extract::<f64>() {
            other_f
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Unsupported operand type for **"));
        };
        
        let res = self.0.powf(exp);
        if res.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err("Math domain error in Scalar power"));
        }
        Ok(Scalar(res))
    }

    // --- Unary ---
    pub fn __neg__(&self) -> Scalar {
        Scalar(-self.0)
    }

    pub fn __abs__(&self) -> Scalar {
        Scalar(self.0.abs())
    }
}

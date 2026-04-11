use pyo3::prelude::*;
use crate::scalar::core::Scalar;

#[pymethods]
impl Scalar {
    pub fn sqrt(&self) -> PyResult<Scalar> {
        if self.0 < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Square root of negative Scalar"));
        }
        Ok(Scalar(self.0.sqrt()))
    }

    pub fn sin(&self) -> Scalar {
        Scalar(self.0.sin())
    }

    pub fn cos(&self) -> Scalar {
        Scalar(self.0.cos())
    }

    pub fn tan(&self) -> Scalar {
        Scalar(self.0.tan())
    }

    pub fn exp(&self) -> Scalar {
        Scalar(self.0.exp())
    }

    #[pyo3(signature = (base=None))]
    pub fn log(&self, base: Option<f64>) -> PyResult<Scalar> {
        if self.0 <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Log of non-positive Scalar"));
        }
        match base {
            Some(b) => {
                if b <= 0.0 || b == 1.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("Invalid base for Scalar log"));
                }
                Ok(Scalar(self.0.log(b)))
            }
            None => Ok(Scalar(self.0.ln())),
        }
    }

    pub fn ceil(&self) -> Scalar {
        Scalar(self.0.ceil())
    }

    pub fn floor(&self) -> Scalar {
        Scalar(self.0.floor())
    }

    pub fn round(&self) -> Scalar {
        Scalar(self.0.round())
    }

    pub fn abs(&self) -> Scalar {
        Scalar(self.0.abs())
    }

    pub fn pow(&self, exp: f64) -> PyResult<Scalar> {
        let res = self.0.powf(exp);
        if res.is_nan() {
            return Err(pyo3::exceptions::PyValueError::new_err("Math domain error in Scalar.pow"));
        }
        Ok(Scalar(res))
    }
}

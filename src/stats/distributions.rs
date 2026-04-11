use pyo3::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Discrete, DiscreteCDF};
use statrs::distribution::{Normal as SNormal, StudentsT as SStudentT, Poisson as SPoisson, Exp as SExp};
use statrs::statistics::Distribution;

/// Normal (Gaussian) distribution.
#[pyclass]
pub struct Normal {
    inner: SNormal,
}

#[pymethods]
impl Normal {
    /// Create a new Normal distribution with mean `mu` and standard deviation `sigma`.
    #[new]
    pub fn new(mu: f64, sigma: f64) -> PyResult<Self> {
        match SNormal::new(mu, sigma) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid Normal params: {}", e))),
        }
    }
    /// Probability density function at `x`.
    pub fn pdf(&self, x: f64) -> f64 { self.inner.pdf(x) }
    /// Cumulative distribution function at `x`.
    pub fn cdf(&self, x: f64) -> f64 { self.inner.cdf(x) }
    /// Percent point function (inverse CDF) at `x`.
    pub fn ppf(&self, x: f64) -> f64 { self.inner.inverse_cdf(x) }
    /// Theoretical mean of the distribution.
    pub fn mean(&self) -> f64 { self.inner.mean().unwrap_or(f64::NAN) }
}

/// Student's T-distribution.
#[pyclass]
pub struct StudentT {
    inner: SStudentT,
}

#[pymethods]
impl StudentT {
    /// Create a new Student's T-distribution.
    #[new]
    pub fn new(location: f64, scale: f64, freedom: f64) -> PyResult<Self> {
        match SStudentT::new(location, scale, freedom) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid StudentT params: {}", e))),
        }
    }
    pub fn pdf(&self, x: f64) -> f64 { self.inner.pdf(x) }
    pub fn cdf(&self, x: f64) -> f64 { self.inner.cdf(x) }
    pub fn ppf(&self, x: f64) -> f64 { self.inner.inverse_cdf(x) }
}

/// Poisson distribution.
#[pyclass]
pub struct Poisson {
    inner: SPoisson,
}

#[pymethods]
impl Poisson {
    /// Create a new Poisson distribution with intensity `lambda`.
    #[new]
    pub fn new(lambda: f64) -> PyResult<Self> {
        match SPoisson::new(lambda) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid Poisson params: {}", e))),
        }
    }
    /// Probability mass function at `k`.
    pub fn pmf(&self, k: u64) -> f64 { self.inner.pmf(k) }
    /// Cumulative distribution function at `k`.
    pub fn cdf(&self, k: f64) -> f64 { self.inner.cdf(k.floor() as u64) }
}

/// Exponential distribution.
#[pyclass]
pub struct Exponential {
    inner: SExp,
}

#[pymethods]
impl Exponential {
    /// Create a new Exponential distribution with `rate`.
    #[new]
    pub fn new(rate: f64) -> PyResult<Self> {
        match SExp::new(rate) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid Exp params: {}", e))),
        }
    }
    pub fn pdf(&self, x: f64) -> f64 { self.inner.pdf(x) }
    pub fn cdf(&self, x: f64) -> f64 { self.inner.cdf(x) }
    pub fn ppf(&self, x: f64) -> f64 { self.inner.inverse_cdf(x) }
}

pub fn register_distributions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Normal>()?;
    m.add_class::<StudentT>()?;
    m.add_class::<Poisson>()?;
    m.add_class::<Exponential>()?;
    Ok(())
}

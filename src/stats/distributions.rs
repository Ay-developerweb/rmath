use pyo3::prelude::*;
use rayon::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Discrete, DiscreteCDF};
use statrs::distribution::{Normal as SNormal, StudentsT as SStudentT, Poisson as SPoisson, Exp as SExp};
use statrs::statistics::Distribution;

const PAR_THRESHOLD: usize = 8_192;

/// Normal (Gaussian) distribution: N(μ, σ).
///
/// The probability density function is given by:
/// f(x) = (1 / (σ√(2π))) * exp(-0.5 * ((x - μ) / σ)²)
///
/// Examples:
///     >>> from rmath.stats import Normal
///     >>> dist = Normal(mu=0.0, sigma=1.0)
///     >>> dist.pdf(0.0)
///     0.3989...
///     >>> dist.sample(5)
///     Vector([0.1234, -0.5678, ...])
#[pyclass(module = "rmath.stats")]
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
    /// Cumulative distribution function at `x` (P(X <= x)).
    pub fn cdf(&self, x: f64) -> f64 { self.inner.cdf(x) }
    /// Percent point function (inverse CDF) at `x`.
    pub fn ppf(&self, x: f64) -> f64 { self.inner.inverse_cdf(x) }
    /// Theoretical mean of the distribution.
    pub fn mean(&self) -> f64 { self.inner.mean().unwrap_or(f64::NAN) }
    
    /// Sample `n` values from the distribution into a high-performance Vector.
    ///
    /// Parallelized via Rayon for large samples (n >= 8192).
    pub fn sample(&self, n: usize) -> crate::vector::Vector {
        let dist = self.inner;
        let data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter()
                .map_init(|| rand::thread_rng(), move |rng, _| {
                    rand::distributions::Distribution::sample(&dist, rng)
                })
                .collect()
        } else {
            let mut rng = rand::thread_rng();
            (0..n).map(|_| rand::distributions::Distribution::sample(&dist, &mut rng)).collect()
        };
        crate::vector::Vector::new(data)
    }
}

/// Student's T-distribution: t(ν).
///
/// Used for hypothesis testing when the sample size is small and the 
/// population standard deviation is unknown.
///
/// Examples:
///     >>> from rmath.stats import StudentT
///     >>> dist = StudentT(location=0.0, scale=1.0, freedom=10.0)
///     >>> dist.cdf(1.0)
///     0.8206...
#[pyclass(module = "rmath.stats")]
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
    /// Sample `n` values from the distribution into a high-performance Vector.
    ///
    /// Parallelized via Rayon for large samples (n >= 8192).
    pub fn sample(&self, n: usize) -> crate::vector::Vector {
        let dist = self.inner;
        let data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter()
                .map_init(|| rand::thread_rng(), move |rng, _| {
                    rand::distributions::Distribution::sample(&dist, rng)
                })
                .collect()
        } else {
            let mut rng = rand::thread_rng();
            (0..n).map(|_| rand::distributions::Distribution::sample(&dist, &mut rng)).collect()
        };
        crate::vector::Vector::new(data)
    }
}

/// Poisson distribution: Pois(λ).
///
/// Models the number of events occurring in a fixed interval of time 
/// or space.
///
/// Examples:
///     >>> from rmath.stats import Poisson
///     >>> dist = Poisson(lambda_=5.0)
///     >>> dist.pmf(5)
///     0.1754...
#[pyclass(module = "rmath.stats")]
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
    /// Sample `n` values from the distribution into a high-performance Vector.
    ///
    /// Parallelized via Rayon for large samples (n >= 8192).
    pub fn sample(&self, n: usize) -> crate::vector::Vector {
        let dist = self.inner;
        let data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter()
                .map_init(|| rand::thread_rng(), move |rng, _| {
                    rand::distributions::Distribution::sample(&dist, rng)
                })
                .collect()
        } else {
            let mut rng = rand::thread_rng();
            (0..n).map(|_| rand::distributions::Distribution::sample(&dist, &mut rng)).collect()
        };
        crate::vector::Vector::new(data)
    }
}

/// Exponential distribution: Exp(λ).
///
/// Models the time between events in a Poisson point process.
///
/// Examples:
///     >>> from rmath.stats import Exponential
///     >>> dist = Exponential(rate=2.0)
///     >>> dist.pdf(0.5)
///     0.7357...
#[pyclass(module = "rmath.stats")]
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
    /// Sample `n` values from the distribution into a high-performance Vector.
    ///
    /// Parallelized via Rayon for large samples (n >= 8192).
    pub fn sample(&self, n: usize) -> crate::vector::Vector {
        let dist = self.inner;
        let data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter()
                .map_init(|| rand::thread_rng(), move |rng, _| {
                    rand::distributions::Distribution::sample(&dist, rng)
                })
                .collect()
        } else {
            let mut rng = rand::thread_rng();
            (0..n).map(|_| rand::distributions::Distribution::sample(&dist, &mut rng)).collect()
        };
        crate::vector::Vector::new(data)
    }
}

pub fn register_distributions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Normal>()?;
    m.add_class::<StudentT>()?;
    m.add_class::<Poisson>()?;
    m.add_class::<Exponential>()?;
    Ok(())
}

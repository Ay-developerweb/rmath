use pyo3::prelude::*;
use ::rand::prelude::*;
use ::rand::rngs::StdRng;
use ::rand::SeedableRng;
use ::rand_distr::{StandardNormal, Distribution};
use super::core::Vector;

// ---------------------------------------------------------------------------
// Random constructors
// ---------------------------------------------------------------------------

/// Uniform random vector in [0, 1).
///
/// Examples:
///     >>> from rmath.vector import rand
///     >>> v = rand(100)
#[pyfunction(name = "rand")]
pub fn vector_rand(n: usize) -> Vector {
    let mut rng = ::rand::thread_rng();
    Vector::new((0..n).map(|_| rng.r#gen::<f64>()).collect())
}

/// Standard-normal random vector (mean=0, std=1).
///
/// Examples:
///     >>> from rmath.vector import randn
///     >>> v = randn(100)
#[pyfunction(name = "randn")]
pub fn vector_randn(n: usize) -> Vector {
    let mut rng = ::rand::thread_rng();
    let dist = StandardNormal;
    Vector::new((0..n).map(|_| dist.sample(&mut rng)).collect())
}

/// Seeded uniform random vector — reproducible.
/// Uses StdRng (ChaCha-based, available in rand 0.8 without extra features).
#[pyfunction]
pub fn rand_seeded(n: usize, seed: u64) -> Vector {
    vector_rand_seeded(n, seed)
}

/// Seeded standard-normal random vector — reproducible.
#[pyfunction]
pub fn randn_seeded(n: usize, seed: u64) -> Vector {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    Vector::new((0..n).map(|_| dist.sample(&mut rng)).collect())
}

/// Internal — callable from core.rs without PyO3 overhead.
pub fn vector_rand_seeded(n: usize, seed: u64) -> Vector {
    let mut rng = StdRng::seed_from_u64(seed);
    Vector::new((0..n).map(|_| rng.r#gen::<f64>()).collect())
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

/// Create a Vector with a range of values.
///
/// Usage: `arange(stop)` or `arange(start, stop, step)`.
///
/// Examples:
///     >>> from rmath.vector import arange
///     >>> arange(1, 5)
///     Vector([1.0, 2.0, 3.0, 4.0])
#[pyfunction]
#[pyo3(signature = (start, stop = None, step = 1.0))]
pub fn arange(start: f64, stop: Option<f64>, step: f64) -> PyResult<Vector> {
    if step == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("step cannot be zero"));
    }
    Ok(Vector::arange(start, stop, step))
}

#[pyfunction(name = "range")]
#[pyo3(signature = (start, stop = None, step = 1.0))]
pub fn vector_range(start: f64, stop: Option<f64>, step: f64) -> PyResult<Vector> {
    arange(start, stop, step)
}

/// Create a Vector with `num` evenly spaced values.
#[pyfunction]
pub fn linspace(start: f64, stop: f64, num: usize) -> Vector {
    Vector::linspace(start, stop, num)
}

#[pyfunction]
pub fn zeros(n: usize) -> Vector { Vector::zeros_internal(n) }

#[pyfunction]
pub fn ones(n: usize) -> Vector { Vector::ones_internal(n) }

#[pyfunction]
pub fn full(n: usize, val: f64) -> Vector { Vector::full_internal(n, val) }

#[pyfunction]
#[pyo3(signature = (start_or_stop, stop = None, step = 1.0))]
pub fn sum_range(start_or_stop: f64, stop: Option<f64>, step: f64) -> f64 {
    Vector::sum_range(start_or_stop, stop, step)
}

// ---------------------------------------------------------------------------
// Free-function math wrappers (module-level functional style)
// ---------------------------------------------------------------------------

#[pyfunction] pub fn sin(v: &Vector)  -> Vector { v.sin() }
#[pyfunction] pub fn cos(v: &Vector)  -> Vector { v.cos() }
#[pyfunction] pub fn tan(v: &Vector)  -> Vector { v.tan() }
#[pyfunction] pub fn asin(v: &Vector) -> Vector { v.asin() }
#[pyfunction] pub fn acos(v: &Vector) -> Vector { v.acos() }
#[pyfunction] pub fn atan(v: &Vector) -> Vector { v.atan() }
#[pyfunction] pub fn sinh(v: &Vector) -> Vector { v.sinh() }
#[pyfunction] pub fn cosh(v: &Vector) -> Vector { v.cosh() }
#[pyfunction] pub fn tanh(v: &Vector) -> Vector { v.tanh() }
#[pyfunction] pub fn exp(v: &Vector)  -> Vector { v.exp() }
#[pyfunction] pub fn exp2(v: &Vector) -> Vector { v.exp2() }
#[pyfunction] pub fn expm1(v: &Vector)-> Vector { v.expm1() }
#[pyfunction] pub fn log(v: &Vector)  -> Vector { v.log() }
#[pyfunction] pub fn log2(v: &Vector) -> Vector { v.log2() }
#[pyfunction] pub fn log10(v: &Vector)-> Vector { v.log10() }
#[pyfunction] pub fn log1p(v: &Vector)-> Vector { v.log1p() }
#[pyfunction] pub fn sqrt(v: &Vector) -> Vector { v.sqrt() }
#[pyfunction] pub fn cbrt(v: &Vector) -> Vector { v.cbrt() }
#[pyfunction] pub fn abs(v: &Vector)  -> Vector { v.abs() }
#[pyfunction] pub fn ceil(v: &Vector) -> Vector { v.ceil() }
#[pyfunction] pub fn floor(v: &Vector)-> Vector { v.floor() }
#[pyfunction] pub fn round(v: &Vector)-> Vector { v.round() }
#[pyfunction] pub fn trunc(v: &Vector)-> Vector { v.trunc() }
#[pyfunction] pub fn fract(v: &Vector)-> Vector { v.fract() }
#[pyfunction] pub fn signum(v: &Vector)->Vector { v.signum() }
#[pyfunction] pub fn recip(v: &Vector)-> Vector { v.recip() }

// ---------------------------------------------------------------------------
// Free-function reducers
// ---------------------------------------------------------------------------

#[pyfunction] pub fn vsum(v: &Vector) -> f64 { v.sum() }
#[pyfunction] pub fn prod(v: &Vector) -> f64 { v.prod() }

/// Calculate the arithmetic mean of a vector or sequence.
#[pyfunction]
pub fn mean(v: &Vector) -> PyResult<f64> {
    let m = v.mean();
    if m.is_nan() {
        Err(pyo3::exceptions::PyValueError::new_err("mean of empty vector"))
    } else {
        Ok(m)
    }
}

#[pyfunction] pub fn variance(v: &Vector) -> f64 { v.variance() }
#[pyfunction] pub fn pop_variance(v: &Vector) -> f64 { v.pop_variance() }
#[pyfunction] pub fn std_dev(v: &Vector) -> f64 { v.std_dev() }
#[pyfunction] pub fn pop_std_dev(v: &Vector) -> f64 { v.pop_std_dev() }
#[pyfunction] pub fn median(v: &Vector) -> f64 { v.median() }
#[pyfunction] pub fn percentile(v: &Vector, q: f64) -> PyResult<f64> { v.percentile(q) }

#[pyfunction]
pub fn vmin(v: &Vector) -> PyResult<f64> {
    let m = v.min();
    if m.is_nan() { Err(pyo3::exceptions::PyValueError::new_err("min of empty vector")) } else { Ok(m) }
}

#[pyfunction]
pub fn vmax(v: &Vector) -> PyResult<f64> {
    let m = v.max();
    if m.is_nan() { Err(pyo3::exceptions::PyValueError::new_err("max of empty vector")) } else { Ok(m) }
}

#[pyfunction] pub fn argmin(v: &Vector) -> isize { v.argmin() }
#[pyfunction] pub fn argmax(v: &Vector) -> isize { v.argmax() }
#[pyfunction] pub fn norm(v: &Vector) -> f64 { v.norm() }
#[pyfunction] pub fn norm_l1(v: &Vector) -> f64 { v.norm_l1() }
#[pyfunction] pub fn norm_inf(v: &Vector) -> f64 { v.norm_inf() }
#[pyfunction] pub fn norm_lp(v: &Vector, p: f64) -> PyResult<f64> { v.norm_lp(p) }
#[pyfunction] pub fn dot(v1: &Vector, v2: &Vector) -> PyResult<f64> { v1.dot(v2) }

// ---------------------------------------------------------------------------
// Free-function arithmetic
// ---------------------------------------------------------------------------

#[pyfunction] pub fn add_scalar(v: &Vector, s: f64) -> Vector { v.map_internal(|x| x + s) }
#[pyfunction] pub fn sub_scalar(v: &Vector, s: f64) -> Vector { v.map_internal(|x| x - s) }
#[pyfunction] pub fn mul_scalar(v: &Vector, s: f64) -> Vector { v.map_internal(|x| x * s) }
#[pyfunction]
pub fn div_scalar(v: &Vector, s: f64) -> PyResult<Vector> {
    if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero")); }
    Ok(v.map_internal(|x| x / s))
}
#[pyfunction] pub fn pow_scalar(v: &Vector, e: f64) -> Vector { v.pow_scalar(e) }
#[pyfunction] pub fn clamp(v: &Vector, lo: f64, hi: f64) -> Vector { v.clamp(lo, hi) }

#[pyfunction] pub fn add_vec(v1: &Vector, v2: &Vector) -> PyResult<Vector> { v1.zip_map_internal(v2, |a,b| a+b) }
#[pyfunction] pub fn sub_vec(v1: &Vector, v2: &Vector) -> PyResult<Vector> { v1.zip_map_internal(v2, |a,b| a-b) }
#[pyfunction] pub fn mul_vec(v1: &Vector, v2: &Vector) -> PyResult<Vector> { v1.zip_map_internal(v2, |a,b| a*b) }
#[pyfunction]
pub fn div_vec(v1: &Vector, v2: &Vector) -> PyResult<Vector> {
    // Documents IEEE behaviour: 0-divisors yield INFINITY, not an error.
    v1.zip_map_internal(v2, |a, b| a / b)
}

// ---------------------------------------------------------------------------
// Free-function predicates
// ---------------------------------------------------------------------------

#[pyfunction] pub fn isnan(v: &Vector)     -> Vec<bool> { v.isnan() }
#[pyfunction] pub fn isfinite(v: &Vector)  -> Vec<bool> { v.isfinite() }
#[pyfunction] pub fn isinf(v: &Vector)     -> Vec<bool> { v.isinf() }
#[pyfunction] pub fn is_integer(v: &Vector)-> Vec<bool> { v.is_integer() }
#[pyfunction] pub fn is_prime(v: &Vector)  -> Vec<bool> { v.is_prime() }
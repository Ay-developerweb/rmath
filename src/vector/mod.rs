pub mod core;
pub mod ops;
pub mod complex;

pub use core::{Vector, VectorIter};
// pub(crate) use ops::{vector_rand, vector_randn, vector_rand_seeded};

use pyo3::prelude::*;

/// Register the entire vector subsystem into a Python submodule.
///
/// Exposes:
///   Classes  : Vector, VectorIter
///   Constructors: zeros, ones, full, arange, linspace, rand, randn,
///                 rand_seeded, randn_seeded, sum_range
///   Math fns : sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
///              exp, exp2, expm1, log, log2, log10, log1p, sqrt, cbrt,
///              abs, ceil, floor, round, trunc, fract, signum, recip
///   Reducers : vsum, prod, mean, variance, pop_variance, std_dev,
///              pop_std_dev, median, percentile, vmin, vmax, argmin,
///              argmax, norm, norm_l1, norm_inf, norm_lp, dot
///   Arithmetic: add_scalar, sub_scalar, mul_scalar, div_scalar,
///               pow_scalar, clamp, add_vec, sub_vec, mul_vec, div_vec
///   Predicates: isnan, isfinite, isinf, is_integer, is_prime
pub fn register_vector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", "rmath.vector — high-performance parallelized element-wise math.

This module provides the `Vector` class and functional wrappers for performing 
massively parallel mathematical operations on 1D numeric data. 

Key Features:
    1. Parallel Execution: Operations on large vectors (>8,192 elements) 
       automatically use the Rayon thread pool, bypassing the Python GIL.
       Thread safety is guaranteed via Rust's Send/Sync traits.
       
    2. Tiered Storage: 
       - Inline: Small vectors (<32 elements) use zero-allocation stack storage.
       - Heap: Large vectors use Arc<Vec<f64>> for efficient sharing and COW mutation.
       
    3. Loop Fusion: The `.lazy()` method allows bridging into `LazyPipeline` 
       for complex chains, fusing multiple maps and filters into a single pass.

Usage:
    >>> import rmath.vector as rv
    >>> v = rv.Vector([1, 2, 3])
    >>> (v * 2.0).sum()
    12.0
")?;

    // --- Classes ---
    m.add_class::<Vector>()?;
    m.add_class::<VectorIter>()?;
    m.add_class::<complex::ComplexVector>()?;

    // --- Constructors ---
    m.add_function(wrap_pyfunction!(ops::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ones, m)?)?;
    m.add_function(wrap_pyfunction!(ops::full, m)?)?;
    m.add_function(wrap_pyfunction!(ops::arange, m)?)?;
    m.add_function(wrap_pyfunction!(ops::vector_range, m)?)?;
    m.add_function(wrap_pyfunction!(ops::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sum_range, m)?)?;
    m.add_function(wrap_pyfunction!(ops::vector_rand, m)?)?;
    m.add_function(wrap_pyfunction!(ops::vector_randn, m)?)?;
    m.add_function(wrap_pyfunction!(ops::rand_seeded, m)?)?;
    m.add_function(wrap_pyfunction!(ops::randn_seeded, m)?)?;

    // --- Math ---
    m.add_function(wrap_pyfunction!(ops::sin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cos, m)?)?;
    m.add_function(wrap_pyfunction!(ops::tan, m)?)?;
    m.add_function(wrap_pyfunction!(ops::asin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::acos, m)?)?;
    m.add_function(wrap_pyfunction!(ops::atan, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sinh, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cosh, m)?)?;
    m.add_function(wrap_pyfunction!(ops::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(ops::exp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::exp2, m)?)?;
    m.add_function(wrap_pyfunction!(ops::expm1, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log2, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log10, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log1p, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cbrt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::abs, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ceil, m)?)?;
    m.add_function(wrap_pyfunction!(ops::floor, m)?)?;
    m.add_function(wrap_pyfunction!(ops::round, m)?)?;
    m.add_function(wrap_pyfunction!(ops::trunc, m)?)?;
    m.add_function(wrap_pyfunction!(ops::fract, m)?)?;
    m.add_function(wrap_pyfunction!(ops::signum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::recip, m)?)?;
    m.add_function(wrap_pyfunction!(ops::dot, m)?)?;

    // --- Reducers ---
    m.add_function(wrap_pyfunction!(ops::vsum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::prod, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mean, m)?)?;
    m.add_function(wrap_pyfunction!(ops::variance, m)?)?;
    m.add_function(wrap_pyfunction!(ops::pop_variance, m)?)?;
    m.add_function(wrap_pyfunction!(ops::std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(ops::pop_std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(ops::median, m)?)?;
    m.add_function(wrap_pyfunction!(ops::percentile, m)?)?;
    m.add_function(wrap_pyfunction!(ops::vmin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::vmax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::argmin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::argmax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::norm, m)?)?;
    m.add_function(wrap_pyfunction!(ops::norm_l1, m)?)?;
    m.add_function(wrap_pyfunction!(ops::norm_inf, m)?)?;
    m.add_function(wrap_pyfunction!(ops::norm_lp, m)?)?;

    // --- Arithmetic ---
    m.add_function(wrap_pyfunction!(ops::add_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sub_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mul_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(ops::div_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(ops::pow_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(ops::clamp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::add_vec, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sub_vec, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mul_vec, m)?)?;
    m.add_function(wrap_pyfunction!(ops::div_vec, m)?)?;

    // --- Predicates ---
    m.add_function(wrap_pyfunction!(ops::isnan, m)?)?;
    m.add_function(wrap_pyfunction!(ops::isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(ops::isinf, m)?)?;
    m.add_function(wrap_pyfunction!(ops::is_integer, m)?)?;
    m.add_function(wrap_pyfunction!(ops::is_prime, m)?)?;

    Ok(())
}
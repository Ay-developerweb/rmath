pub mod core;
pub mod ops;
pub mod math;
pub mod pipeline;
pub mod complex;

pub use core::Scalar;
pub use complex::Complex;
pub use pipeline::{LazyPipeline, loop_range, from_list, zeros, linspace, from_shared_buffer};

use pyo3::prelude::*;

/// Register the entire scalar subsystem into a Python submodule.
///
/// Exposes:
///   Classes  : Scalar, Complex, LazyPipeline
///   Functions: loop_range, from_list, zeros, linspace
///   Constants: pi, e, tau, inf, nan, sqrt2, ln2, ln10
pub fn register_scalar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", "rmath.scalar — high-performance scalar math operations backed by Rust.

This module provides Rust-accelerated scalar types and the LazyPipeline for loop fusion.
By wrapping 64-bit IEEE-754 floats in a native Rust container, we enable math operations 
that release the Python GIL and achieve significantly higher performance than pure Python 
loops.

Performance Philosophy:
    Rmath scalar operations are designed for high-throughput numeric workloads. 
    While Python's built-in `float` is highly optimized for general-purpose use, 
    `rmath.Scalar` allows for chaining operations that execute entirely in Rust, 
    avoiding the overhead of the Python interpreter for every intermediate step.

NaN and Error Policy:
    Rmath distinguishes between *operators* and *named methods*:
    
    1. Operators (+, -, *, /, **) follow IEEE-754 semantics strictly. They 
       never raise exceptions for domain errors; they return `NaN` or `inf` to 
       maintain interoperability with Python's numeric tower.
       
    2. Named Methods (.sqrt(), .log(), .pow()) are stricter. Since they represent 
       a deliberate mathematical call, they raise `ValueError` for domain 
       errors to help catch bugs early in the pipeline.

Import style::

    import rmath.scalar as rs
    s = rs.Scalar(1.0)
    result = s.sin().cos().exp()
")?;

    // --- Classes ---
    m.add_class::<Scalar>()?;
    m.add_class::<Complex>()?;
    m.add_class::<LazyPipeline>()?;

    // --- Free functions ---
    m.add_function(wrap_pyfunction!(loop_range, m)?)?;
    m.add_function(wrap_pyfunction!(from_list, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;

    // --- Mathematical constants as Scalar values ---
    //
    // Exposing these as Scalar (not raw f64) means the user can feed them
    // directly into Scalar arithmetic without boxing:
    //   result = sc.Scalar(2.0) * sc.pi   # stays native throughout
    m.add("pi",    Scalar::new(std::f64::consts::PI))?;
    m.add("e",     Scalar::new(std::f64::consts::E))?;
    m.add("tau",   Scalar::new(std::f64::consts::TAU))?;
    m.add("sqrt2", Scalar::new(std::f64::consts::SQRT_2))?;
    m.add("ln2",   Scalar::new(std::f64::consts::LN_2))?;
    m.add("ln10",  Scalar::new(std::f64::consts::LN_10))?;

    // --- Special float values as Scalar ---
    m.add("inf",   Scalar::new(f64::INFINITY))?;
    m.add("nan",   Scalar::new(f64::NAN))?;

    Ok(())
}
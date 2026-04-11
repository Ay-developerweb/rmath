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
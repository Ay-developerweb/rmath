pub mod core;
pub mod ops;
pub mod math;
pub mod pipeline;
pub mod complex;

pub use core::Scalar;
pub use pipeline::{LazyPipeline, loop_range, from_shared_buffer};
pub use complex::Complex;

use pyo3::prelude::*;

/// Registers the new modular scalar system.
pub fn register_scalar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Scalar>()?;
    m.add_class::<Complex>()?;
    m.add_class::<LazyPipeline>()?;
    m.add_function(wrap_pyfunction!(loop_range, m)?)?;
    
    // We will export functions here that can work on both 
    // raw floats and Scalar objects for maximum flexibility.
    
    Ok(())
}

use pyo3::prelude::*;

pub mod dual;
pub mod integration;
pub mod solver;

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<dual::Dual>()?;
    m.add_function(wrap_pyfunction!(integration::integrate_trapezoidal, m)?)?;
    m.add_function(wrap_pyfunction!(integration::integrate_simpson, m)?)?;
    m.add_function(wrap_pyfunction!(integration::integrate_simpson_array, m)?)?;
    m.add_function(wrap_pyfunction!(solver::find_root_newton, m)?)?;
    Ok(())
}

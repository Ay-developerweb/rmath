use pyo3::prelude::*;

pub mod dual;
pub mod integration;
pub mod solver;

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr(
        "__doc__",
        "rmath.calculus — Advanced automatic differentiation and numerical integration.\n\n\
        This module provides tools for exact derivative computation using Forward-Mode \
        Automatic Differentiation (Dual numbers) and high-precision numerical integration \
        (Trapezoidal, Simpson's rules).\n\n\
        Features:\n\
            - `Dual`: Forward-mode AD for exact analytical gradients without reverse-tape overhead.\n\
            - `integrate_trapezoidal`, `integrate_simpson`: Massively parallel integration over 1D arrays.\n\
            - `find_root_newton`: Robust root-finding algorithms.",
    )?;
    
    m.add_class::<dual::Dual>()?;
    m.add_function(wrap_pyfunction!(integration::integrate_trapezoidal, m)?)?;
    m.add_function(wrap_pyfunction!(integration::integrate_simpson, m)?)?;
    m.add_function(wrap_pyfunction!(integration::integrate_simpson_array, m)?)?;
    m.add_function(wrap_pyfunction!(solver::find_root_newton, m)?)?;
    Ok(())
}

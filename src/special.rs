use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vector::Vector;
use statrs::function::gamma as s_gamma_internal;
use statrs::function::erf as s_erf_internal;

/// Helper to map a special function across a Vector or list (Standard Vec).
fn map_special<'py, F>(data: Bound<'py, PyAny>, op: F) -> PyResult<Bound<'py, PyAny>> 
where F: Fn(f64) -> f64 + Sync + Send 
{
    let py = data.py();

    // 1. Array Path (High Priority)
    if let Ok(a) = data.extract::<PyRef<crate::array::Array>>() {
        let new_a = a.map_elements(op);
        return Ok(Bound::new(py, new_a)?.into_any());
    }

    // 2. Vector Path
    if let Ok(v) = data.extract::<PyRef<Vector>>() {
        let new_v = v.map_internal(op);
        return Ok(Bound::new(py, new_v)?.into_any());
    }
    
    // 3. Scalar Path
    if let Ok(x) = data.extract::<f64>() {
        return Ok(op(x).into_pyobject(py)?.into_any());
    }
    
    // 4. List Fallback
    let list: Vec<f64> = data.extract()?;
    let res: Vec<f64> = if list.len() >= crate::array::core::PAR_THRESHOLD {
        list.par_iter().map(|&x| op(x)).collect()
    } else {
        list.iter().map(|&x| op(x)).collect()
    };
    let new_v = Vector::new(res);
    Ok(Bound::new(py, new_v)?.into_any())
}

/// Compute the Gamma function Γ(x).
///
/// Supports broadcasting across Scalars, Vectors, and Arrays.
///
/// Examples:
///     >>> from rmath import special
///     >>> special.gamma(5.0)  # Γ(5) = 4! = 24
///     24.0
#[pyfunction(name = "gamma")]
pub fn rust_gamma<'py>(data: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    map_special(data, s_gamma_internal::gamma)
}

/// Compute the natural logarithm of the Gamma function, ln|Γ(x)|.
///
/// Faster and more numerically stable than calling `ln(gamma(x))`.
///
/// Examples:
///     >>> from rmath import special
///     >>> special.ln_gamma(10.0)
///     12.8018...
#[pyfunction(name = "ln_gamma")]
pub fn rust_ln_gamma<'py>(data: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    map_special(data, s_gamma_internal::ln_gamma)
}

/// Compute the Error function erf(x).
///
/// The Error function is defined as: 
/// erf(x) = (2/√π) * ∫ exp(-t²) dt from 0 to x.
///
/// Examples:
///     >>> from rmath import special
///     >>> special.erf(0.0)
///     0.0
///     >>> special.erf(1.0)
///     0.8427...
#[pyfunction(name = "erf")]
pub fn rust_erf<'py>(data: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    map_special(data, s_erf_internal::erf)
}

pub fn register_special(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ln_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(rust_erf, m)?)?;
    Ok(())
}

use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vector::Vector;

#[pyfunction]
pub fn euclidean_distance(a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<f64> {
    if let (Ok(v1), Ok(v2)) = (a.extract::<PyRef<Vector>>(), b.extract::<PyRef<Vector>>()) {
        if v1.len_internal() != v2.len_internal() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
        let sum: f64 = v1.with_slice(|s1| {
            v2.with_slice(|s2| {
                s1.par_iter().zip(s2.par_iter()).map(|(x, y)| (x - y).powi(2)).sum()
            })
        });
        return Ok(sum.sqrt());
    }
    let v1: Vec<f64> = a.extract()?;
    let v2: Vec<f64> = b.extract()?;
    if v1.len() != v2.len() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
    let sum: f64 = v1.par_iter().zip(v2.par_iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Ok(sum.sqrt())
}

#[pyfunction]
pub fn manhattan_distance(a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<f64> {
    if let (Ok(v1), Ok(v2)) = (a.extract::<PyRef<Vector>>(), b.extract::<PyRef<Vector>>()) {
        if v1.len_internal() != v2.len_internal() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
        let sum: f64 = v1.with_slice(|s1| {
            v2.with_slice(|s2| {
                s1.par_iter().zip(s2.par_iter()).map(|(x, y)| (x - y).abs()).sum()
            })
        });
        return Ok(sum);
    }
    let v1: Vec<f64> = a.extract()?;
    let v2: Vec<f64> = b.extract()?;
    if v1.len() != v2.len() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
    let sum: f64 = v1.par_iter().zip(v2.par_iter()).map(|(x, y)| (x - y).abs()).sum();
    Ok(sum)
}

#[pyfunction]
pub fn minkowski_distance(a: Bound<'_, PyAny>, b: Bound<'_, PyAny>, p: f64) -> PyResult<f64> {
    if let (Ok(v1), Ok(v2)) = (a.extract::<PyRef<Vector>>(), b.extract::<PyRef<Vector>>()) {
        if v1.len_internal() != v2.len_internal() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
        let sum: f64 = v1.with_slice(|s1| {
            v2.with_slice(|s2| {
                s1.par_iter().zip(s2.par_iter()).map(|(x, y)| (x - y).abs().powf(p)).sum()
            })
        });
        return Ok(sum.powf(1.0 / p));
    }
    let v1: Vec<f64> = a.extract()?;
    let v2: Vec<f64> = b.extract()?;
    if v1.len() != v2.len() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
    let sum: f64 = v1.par_iter().zip(v2.par_iter()).map(|(x, y)| (x - y).abs().powf(p)).sum();
    Ok(sum.powf(1.0 / p))
}

#[pyfunction]
pub fn cosine_similarity(a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<f64> {
     // Fast Path: Vector
     if let (Ok(v1), Ok(v2)) = (a.extract::<PyRef<Vector>>(), b.extract::<PyRef<Vector>>()) {
         if v1.len_internal() != v2.len_internal() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
         let (dot, norm_a, norm_b): (f64, f64, f64) = v1.with_slice(|s1| {
            v2.with_slice(|s2| {
                s1.par_iter().zip(s2.par_iter())
                    .map(|(x, y)| (x * y, x * x, y * y))
                    .reduce(|| (0.0, 0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2))
            })
         });
        if norm_a == 0.0 || norm_b == 0.0 { return Ok(0.0); }
        return Ok(dot / (norm_a.sqrt() * norm_b.sqrt()));
     }
     
     // Fallback: List
     let v1: Vec<f64> = a.extract()?;
     let v2: Vec<f64> = b.extract()?;
     if v1.len() != v2.len() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
     let (dot, norm_a, norm_b): (f64, f64, f64) = v1.par_iter().zip(v2.par_iter())
        .map(|(x, y)| (x * y, x * x, y * y))
        .reduce(|| (0.0, 0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2));
    
    if norm_a == 0.0 || norm_b == 0.0 { return Ok(0.0); }
    Ok(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

pub fn register_geometry(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(manhattan_distance, m)?)?;
    m.add_function(wrap_pyfunction!(minkowski_distance, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    Ok(())
}

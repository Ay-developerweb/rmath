use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vector::core::Vector;
use crate::array::core::Array;

fn extract_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vector> {
    if let Ok(v) = obj.extract::<PyRef<Vector>>() {
        Ok(v.clone())
    } else {
        Ok(Vector::new(obj.extract::<Vec<f64>>()?))
    }
}

/// Calculate the Euclidean distance (L2 norm) between two vectors.
///
/// Examples:
///     >>> from rmath.vector import Vector
///     >>> from rmath.geometry import euclidean_distance
///     >>> euclidean_distance(Vector([0, 0]), Vector([3, 4]))
///     5.0
#[pyfunction]
pub fn euclidean_distance(v1_any: Bound<'_, PyAny>, v2_any: Bound<'_, PyAny>) -> PyResult<f64> {
    let v1 = extract_vector(&v1_any)?;
    let v2 = extract_vector(&v2_any)?;
    v1.with_slice(|s1| v2.with_slice(|s2| {
        if s1.len() != s2.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Dimension mismatch"));
        }
        let sum_sq: f64 = if s1.len() < 1024 {
            s1.iter().zip(s2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
        } else {
            s1.par_iter().zip(s2.par_iter()).map(|(a, b)| (a - b).powi(2)).sum()
        };
        Ok(sum_sq.sqrt())
    }))
}

/// Calculate the Manhattan distance (L1 norm) between two vectors.
#[pyfunction]
pub fn manhattan_distance(v1_any: Bound<'_, PyAny>, v2_any: Bound<'_, PyAny>) -> PyResult<f64> {
    let v1 = extract_vector(&v1_any)?;
    let v2 = extract_vector(&v2_any)?;
    v1.with_slice(|s1| v2.with_slice(|s2| {
        if s1.len() != s2.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Dimension mismatch"));
        }
        let sum: f64 = if s1.len() < 1024 {
            s1.iter().zip(s2.iter()).map(|(a, b)| (a - b).abs()).sum()
        } else {
            s1.par_iter().zip(s2.par_iter()).map(|(a, b)| (a - b).abs()).sum()
        };
        Ok(sum)
    }))
}

/// Calculate the Minkowski distance (Lp norm) between two vectors.
#[pyfunction]
pub fn minkowski_distance(v1_any: Bound<'_, PyAny>, v2_any: Bound<'_, PyAny>, p: f64) -> PyResult<f64> {
    let v1 = extract_vector(&v1_any)?;
    let v2 = extract_vector(&v2_any)?;
    v1.with_slice(|s1| v2.with_slice(|s2| {
        if s1.len() != s2.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Dimension mismatch"));
        }
        let sum: f64 = if s1.len() < 1024 {
            s1.iter().zip(s2.iter()).map(|(a, b)| (a - b).abs().powf(p)).sum()
        } else {
            s1.par_iter().zip(s2.par_iter()).map(|(a, b)| (a - b).abs().powf(p)).sum()
        };
        Ok(sum.powf(1.0 / p))
    }))
}

/// Calculate the cosine similarity between two vectors.
///
/// Returns a value in `[-1, 1]` representing the cosine of the angle between them.
#[pyfunction]
pub fn cosine_similarity(v1_any: Bound<'_, PyAny>, v2_any: Bound<'_, PyAny>) -> PyResult<f64> {
    let v1 = extract_vector(&v1_any)?;
    let v2 = extract_vector(&v2_any)?;
    v1.with_slice(|s1| v2.with_slice(|s2| {
        if s1.len() != s2.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Dimension mismatch"));
        }
        let (dot, norm_a, norm_b): (f64, f64, f64) = if s1.len() < 1024 {
            s1.iter().zip(s2.iter())
                .map(|(x, y)| (x * y, x * x, y * y))
                .fold((0.0, 0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2))
        } else {
            s1.par_iter().zip(s2.par_iter())
                .map(|(x, y)| (x * y, x * x, y * y))
                .reduce(|| (0.0, 0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2))
        };
            
        if norm_a == 0.0 || norm_b == 0.0 { Ok(0.0) }
        else { Ok(dot / (norm_a.sqrt() * norm_b.sqrt())) }
    }))
}

/// Project vector `v` onto target vector `target`.
#[pyfunction]
pub fn projection(v_any: Bound<'_, PyAny>, target_any: Bound<'_, PyAny>) -> PyResult<Vector> {
    let v = extract_vector(&v_any)?;
    let target = extract_vector(&target_any)?;
    let dot_vt = v.dot(&target)?;
    let dot_tt = target.dot(&target)?;
    if dot_tt == 0.0 { Ok(target.clone()) }
    else { 
        let scalar = dot_vt / dot_tt;
        Ok(target.mul_scalar(scalar))
    }
}

/// Calculate the 3D cross product of two vectors.
#[pyfunction]
pub fn cross_product(v1_any: Bound<'_, PyAny>, v2_any: Bound<'_, PyAny>) -> PyResult<Vector> {
    let v1 = extract_vector(&v1_any)?;
    let v2 = extract_vector(&v2_any)?;
    v1.with_slice(|s1| v2.with_slice(|s2| {
        if s1.len() != 3 || s2.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("Cross product is only defined for 3D vectors"));
        }
        let res = vec![
            s1[1] * s2[2] - s1[2] * s2[1],
            s1[2] * s2[0] - s1[0] * s2[2],
            s1[0] * s2[1] - s1[1] * s2[0],
        ];
        Ok(Vector::new(res))
    }))
}

/// Calculate the angle in radians between two vectors.
#[pyfunction]
pub fn angle_between(v1_any: Bound<'_, PyAny>, v2_any: Bound<'_, PyAny>) -> PyResult<f64> {
    let cos_theta = cosine_similarity(v1_any, v2_any)?;
    Ok(cos_theta.clamp(-1.0, 1.0).acos())
}

/// Compute the pairwise Euclidean distance matrix between two sets of points.
///
/// Args:
///     a: First set of points (M x D Array).
///     b: Second set of points (N x D Array).
///
/// Returns:
///     M x N Array containing pairwise Euclidean distances.
#[pyfunction]
pub fn cdist(py: Python<'_>, a: &Array, b: &Array) -> PyResult<Array> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Inputs must be 2D Arrays"));
    }
    let m = a.nrows();
    let n = b.nrows();
    let d = a.ncols();
    if d != b.ncols() {
        return Err(pyo3::exceptions::PyValueError::new_err("Point dimensions must match"));
    }

    py.allow_threads(move || {
        let a_data = a.data();
        let b_data = b.data();
        let mut res_data = vec![0.0; m * n];

        res_data.par_chunks_mut(n).enumerate().for_each(|(i, row_res)| {
            let p1 = &a_data[i * d..(i + 1) * d];
            for j in 0..n {
                let p2 = &b_data[j * d..(j + 1) * d];
                let dist: f64 = p1.iter().zip(p2.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                row_res[j] = dist.sqrt();
            }
        });

        Ok(Array::from_flat(res_data, vec![m, n]))
    })
}

pub mod transforms;
pub mod topology;

pub fn register_geometry(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<transforms::Quaternion>()?;
    m.add_function(wrap_pyfunction!(topology::is_point_in_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(topology::convex_hull, m)?)?;
    
    m.add_function(wrap_pyfunction!(euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(manhattan_distance, m)?)?;
    m.add_function(wrap_pyfunction!(minkowski_distance, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(projection, m)?)?;
    m.add_function(wrap_pyfunction!(cross_product, m)?)?;
    m.add_function(wrap_pyfunction!(angle_between, m)?)?;
    m.add_function(wrap_pyfunction!(cdist, m)?)?;
    Ok(())
}

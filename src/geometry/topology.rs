use pyo3::prelude::*;

/// Determine if a point (x, y) is inside a polygon using the Ray Casting algorithm.
///
/// Returns `True` if the point is inside or on the boundary.
///
/// Examples:
///     >>> from rmath.geometry import is_point_in_polygon
///     >>> poly_x = [0, 2, 2, 0]
///     >>> poly_y = [0, 0, 2, 2]
///     >>> is_point_in_polygon(1, 1, poly_x, poly_y)
///     True
#[pyfunction]
pub fn is_point_in_polygon(x: f64, y: f64, poly_x: Vec<f64>, poly_y: Vec<f64>) -> bool {
    let n = poly_x.len();
    if n != poly_y.len() || n < 3 { return false; }
    
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        if ((poly_y[i] > y) != (poly_y[j] > y)) &&
           (x < (poly_x[j] - poly_x[i]) * (y - poly_y[i]) / (poly_y[j] - poly_y[i]) + poly_x[i]) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

use crate::array::core::Array;

/// Compute the Convex Hull of a set of 2D points using the Monotone Chain algorithm.
///
/// Accepts either an Array of shape (N, 2) or two lists of X and Y coordinates.
///
/// Returns:
///     If input is Array: Returns an Array of shape (M, 2)
///     If input is lists: Returns a tuple of (Hull-X, Hull-Y)
///
/// Examples:
///     >>> from rmath.geometry import convex_hull
///     >>> pts_x = [0, 1, 2, 1, 0]
///     >>> pts_y = [0, 2, 0, 1, 0]
///     >>> hx, hy = convex_hull(pts_x, pts_y)
#[pyfunction]
#[pyo3(signature = (arg1, arg2=None))]
pub fn convex_hull<'py>(py: Python<'py>, arg1: Bound<'py, PyAny>, arg2: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let mut is_array = false;
    let (pts_x, pts_y) = if let Some(a2) = arg2 {
        let x: Vec<f64> = arg1.extract()?;
        let y: Vec<f64> = a2.extract()?;
        (x, y)
    } else {
        let arr: PyRef<Array> = arg1.extract()?;
        if arr.ndim() != 2 || arr.ncols() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Array must be shape (N, 2)"));
        }
        is_array = true;
        let data = arr.data();
        let rows = arr.nrows();
        let mut x = Vec::with_capacity(rows);
        let mut y = Vec::with_capacity(rows);
        for i in 0..rows {
            x.push(data[i*2]);
            y.push(data[i*2+1]);
        }
        (x, y)
    };

    let n = pts_x.len();
    if n != pts_y.len() { return Err(pyo3::exceptions::PyValueError::new_err("Length mismatch")); }
    
    if n < 3 {
        return if is_array {
            let data: Vec<f64> = pts_x.into_iter().zip(pts_y.into_iter()).flat_map(|(x, y)| vec![x, y]).collect();
            Ok(Array::from_flat(data, vec![n, 2]).into_pyobject(py)?.into_any())
        } else {
            Ok(pyo3::types::PyTuple::new(py, vec![pts_x, pts_y])?.into_any())
        };
    }

    let mut pts: Vec<(f64, f64)> = pts_x.into_iter().zip(pts_y.into_iter()).collect();
    // Sort by x, then y
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));

    fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }

    let mut lower = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len()-2], lower[lower.len()-1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len()-2], upper[upper.len()-1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);

    if is_array {
        let m = lower.len();
        let data: Vec<f64> = lower.into_iter().flat_map(|(x, y)| vec![x, y]).collect();
        Ok(Array::from_flat(data, vec![m, 2]).into_pyobject(py)?.into_any())
    } else {
        let (hx, hy): (Vec<f64>, Vec<f64>) = lower.into_iter().unzip();
        Ok(pyo3::types::PyTuple::new(py, vec![hx, hy])?.into_any())
    }
}

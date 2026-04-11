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

/// Compute the Convex Hull of a set of 2D points using the Monotone Chain algorithm.
///
/// Returns a tuple of (Hull-X, Hull-Y) coordinates.
///
/// Examples:
///     >>> from rmath.geometry import convex_hull
///     >>> hx, hy = convex_hull([0, 2, 1, 1], [0, 0, 2, 1])
#[pyfunction]
pub fn convex_hull(points_x: Vec<f64>, points_y: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n = points_x.len();
    if n != points_y.len() { return Err(pyo3::exceptions::PyValueError::new_err("Mismatch")); }
    if n < 3 { return Ok((points_x, points_y)); }

    let mut pts: Vec<(f64, f64)> = points_x.into_iter().zip(points_y.into_iter()).collect();
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

    let (hx, hy): (Vec<f64>, Vec<f64>) = lower.into_iter().unzip();
    Ok((hx, hy))
}

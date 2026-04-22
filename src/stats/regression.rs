use pyo3::prelude::*;
use crate::vector::Vector;
use std::collections::HashMap;

/// Perform simple linear regression on two variables.
///
/// Returns a dictionary containing the slope, intercept, and R-squared values.
/// Calculates the best-fit line using the ordinary least squares (OLS) method.
///
/// Examples:
///     >>> from rmath.stats import linear_regression
///     >>> x = [1, 2, 3, 4, 5]
///     >>> y = [2, 4, 5, 4, 5]
///     >>> res = linear_regression(x, y)
///     >>> res['slope']
///     0.6
#[pyfunction]
pub fn linear_regression(x: Bound<'_, PyAny>, y: Bound<'_, PyAny>) -> PyResult<HashMap<String, f64>> {
    let vx: PyRef<Vector> = x.extract()?;
    let vy: PyRef<Vector> = y.extract()?;
    
    let n = vx.len_internal() as f64;
    if n < 2.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Need at least 2 points for regression"));
    }
    
    // Pearson Correlation coefficient ingredients
    let (mean_x, m2_x, _) = vx.welford();
    let (mean_y, m2_y, _) = vy.welford();
    
    // Covariance (SSxy)
    let ss_xy = vx.with_slice(|sx| vy.with_slice(|sy| {
        sx.iter().zip(sy.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
    }));

    let slope = ss_xy / m2_x;
    let intercept = mean_y - slope * mean_x;
    
    // R-squared
    let ss_total = m2_y;
    let ss_res = vx.with_slice(|sx| vy.with_slice(|sy| {
        sx.iter().zip(sy.iter())
            .map(|(&xi, &yi)| {
                let yi_pred = slope * xi + intercept;
                (yi - yi_pred).powi(2)
            })
            .sum::<f64>()
    }));
    
    let r_sq = 1.0 - (ss_res / ss_total);

    let mut res = HashMap::new();
    res.insert("slope".to_string(), slope);
    res.insert("intercept".to_string(), intercept);
    res.insert("r_squared".to_string(), r_sq);
    Ok(res)
}

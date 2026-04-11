use pyo3::prelude::*;
use crate::vector::core::Vector;

/// Compute the definite integral of `y` with respect to `x` using the trapezoidal rule.
///
/// Examples:
///     >>> from rmath.vector import Vector
///     >>> from rmath.calculus import integrate_trapezoidal
///     >>> x = Vector([0, 1, 2])
///     >>> y = Vector([1, 1, 1])
///     >>> integrate_trapezoidal(x, y)
///     2.0
#[pyfunction]
pub fn integrate_trapezoidal(x: &Vector, y: &Vector) -> PyResult<f64> {
    x.with_slice(|sx| y.with_slice(|sy| {
        let n = sx.len();
        if n != sy.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Length mismatch"));
        }
        if n < 2 { return Ok(0.0); }

        // Area = sum[(x_i+1 - x_i) * (y_i + y_i+1) / 2]
        let area: f64 = sx.windows(2).zip(sy.windows(2))
            .map(|(xw, yw)| (xw[1] - xw[0]) * (yw[0] + yw[1]) * 0.5)
            .sum();
            
        Ok(area)
    }))
}

/// Compute the definite integral of evenly spaced data using Simpson's Rule.
///
/// Uses Simpson's 1/3 rule. For even numbers of points, falls back to the
/// trapezoidal rule for the final interval.
#[pyfunction]
pub fn integrate_simpson_array(y: &Vector, dx: f64) -> PyResult<f64> {
    y.with_slice(|s| {
        let n = s.len();
        if n < 3 { return Err(pyo3::exceptions::PyValueError::new_err("Simpson needs n>=3")); }
        if n % 2 == 0 {
            // Simpson's rule requires odd number of points (even intervals)
            // We use Simpson's 3/8 on the last 4 points or just fallback to trapezoid for last bit
            // For now, let's keep it simple: Simpson's 1/3 + Trapezoid for the last interval
            let mut area = 0.0;
            // Simpson's 1/3 on 0..n-2
            let limit = n - 2;
            for i in (0..limit).step_by(2) {
                area += (dx / 3.0) * (s[i] + 4.0 * s[i+1] + s[i+2]);
            }
            // Trapezoid on last
            area += (dx / 2.0) * (s[n-2] + s[n-1]);
            Ok(area)
        } else {
            let mut area = 0.0;
            for i in (0..n-1).step_by(2) {
                area += (dx / 3.0) * (s[i] + 4.0 * s[i+1] + s[i+2]);
            }
            Ok(area)
        }
    })
}

/// Compute the definite integral of a function `f` from `a` to `b` using Simpson's Rule.
///
/// Requires `n` (number of sub-intervals) to be even.
///
/// Examples:
///     >>> from rmath.calculus import integrate_simpson
///     >>> integrate_simpson(lambda x: x**2, 0, 1, 100)
///     0.3333333333333333
#[pyfunction]
pub fn integrate_simpson(f: PyObject, a: f64, b: f64, n: usize) -> PyResult<f64> {
    if n % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("n must be even for Simpson's Rule"));
    }
    
    Python::with_gil(|py| {
        let dx = (b - a) / (n as f64);
        let mut sum = (f.call1(py, (a,))?.extract::<f64>(py)?) + (f.call1(py, (b,))?.extract::<f64>(py)?);
        
        for i in 1..n {
            let x = a + (i as f64) * dx;
            let val = f.call1(py, (x,))?.extract::<f64>(py)?;
            if i % 2 == 0 {
                sum += 2.0 * val;
            } else {
                sum += 4.0 * val;
            }
        }
        
        Ok(sum * dx / 3.0)
    })
}

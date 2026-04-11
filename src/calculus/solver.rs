use pyo3::prelude::*;
use crate::calculus::dual::Dual;

/// Root finding using Newton's method with Automatic Differentiation.
///
/// This solver uses the derivative information provided by the `Dual` number
/// system to find the roots of a function `f(x) = 0`.
///
/// Args:
///     f: A function that accepts and returns a `rmath.calculus.Dual` number.
///     x0: Initial guess.
///     tol: Convergence tolerance.
///     max_iter: Maximum number of iterations.
///
/// Examples:
///     >>> from rmath.calculus import Dual, find_root_newton
///     >>> f = lambda x: x**2 - 2  # Solve x^2 = 2
///     >>> find_root_newton(f, 1.5, 1e-7, 10)
///     1.4142135623746899
#[pyfunction]
pub fn find_root_newton(f: PyObject, x0: f64, tol: f64, max_iter: usize) -> PyResult<f64> {
    Python::with_gil(|py| {
        let mut x = x0;
        
        for _ in 0..max_iter {
            // Evaluate f(x + epsilon)
            let dual_in = Dual { val: x, der: 1.0 };
            let res = f.call1(py, (dual_in,))?;
            let dual_out: Dual = res.extract(py)?;
            
            let fx = dual_out.val;
            let dfx = dual_out.der;
            
            if fx.abs() < tol {
                return Ok(x);
            }
            
            if dfx == 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err("Derivative is zero, Newton failed"));
            }
            
            x = x - fx / dfx;
        }
        
        Ok(x)
    })
}

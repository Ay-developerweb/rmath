use crate::vector::Vector;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

macro_rules! dispatch_stats_op {
    ($data:expr, $slice:ident, $body:block) => {
        if let Ok(v) = $data.extract::<PyRef<Vector>>() {
            v.with_slice(|$slice| $body)
        } else {
            let owned: Vec<f64> = $data.extract()?;
            let $slice = &owned;
            $body
        }
    };
}

#[pyfunction]
pub fn sum(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, { Ok(slice.par_iter().sum()) })
}

#[pyfunction]
pub fn mean(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mean requires at least one element",
            ));
        }
        let total: f64 = slice.par_iter().sum();
        Ok(total / slice.len() as f64)
    })
}

#[pyfunction]
pub fn variance(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        let n = slice.len();
        if n < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "variance requires at least two elements",
            ));
        }
        let m = slice.par_iter().sum::<f64>() / n as f64;
        let sum_sq: f64 = slice.par_iter().map(|&x| (x - m).powi(2)).sum();
        Ok(sum_sq / (n - 1) as f64)
    })
}

#[pyfunction]
pub fn std_dev(data: Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(variance(data)?.sqrt())
}

#[pyfunction]
pub fn geometric_mean(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "geometric_mean requires at least one element",
            ));
        }
        if slice.iter().any(|x| x.is_nan()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "geometric_mean does not support NaN",
            ));
        }
        if slice.iter().any(|&x| x <= 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "geometric_mean requires all values to be positive",
            ));
        }
        let log_sum: f64 = slice.par_iter().map(|&x| x.ln()).sum();
        Ok((log_sum / slice.len() as f64).exp())
    })
}

#[pyfunction]
pub fn harmonic_mean(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "harmonic_mean requires at least one element",
            ));
        }
        if slice.iter().any(|x| x.is_nan()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "harmonic_mean does not support NaN",
            ));
        }
        if slice.iter().any(|&x| x == 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "harmonic_mean requires all values to be non-zero",
            ));
        }
        let sum_inv: f64 = slice.par_iter().map(|&x| 1.0 / x).sum();
        Ok(slice.len() as f64 / sum_inv)
    })
}

fn _correlation_inner(x: &[f64], y: &[f64]) -> PyResult<f64> {
    let n = x.len();
    if n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "correlation requires at least two elements",
        ));
    }
    if n != y.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "correlation requires x and y to have the same length",
        ));
    }
    let mx = x.par_iter().sum::<f64>() / n as f64;
    let my = y.par_iter().sum::<f64>() / n as f64;
    let (cp, s1, s2): (f64, f64, f64) = x
        .par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| {
            (
                (xi - mx) * (yi - my),
                (xi - mx).powi(2),
                (yi - my).powi(2),
            )
        })
        .reduce(
            || (0.0, 0.0, 0.0),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );
    let den = (s1 * s2).sqrt();
    Ok(if den == 0.0 { 0.0 } else { cp / den })
}

#[pyfunction]
pub fn correlation(x_any: Bound<'_, PyAny>, y_any: Bound<'_, PyAny>) -> PyResult<f64> {
    if let (Ok(v1), Ok(v2)) = (
        x_any.extract::<PyRef<Vector>>(),
        y_any.extract::<PyRef<Vector>>(),
    ) {
        return v1.with_slice(|s1| {
            v2.with_slice(|s2| _correlation_inner(s1, s2))
        });
    }
    let x: Vec<f64> = x_any.extract()?;
    let y: Vec<f64> = y_any.extract()?;
    _correlation_inner(&x, &y)
}

#[pyfunction]
pub fn skewness(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        let n = slice.len();
        if n < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "skewness requires at least three elements",
            ));
        }
        let nf = n as f64;
        let m = slice.par_iter().sum::<f64>() / nf;
        let s = (slice.par_iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
        if s == 0.0 {
            return Ok(0.0);
        }
        // Adjusted Fisher-Pearson standardised moment coefficient
        let m3: f64 = slice.par_iter().map(|&x| ((x - m) / s).powi(3)).sum();
        Ok((nf / ((nf - 1.0) * (nf - 2.0))) * m3)
    })
}

#[pyfunction]
pub fn z_scores(data: Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    dispatch_stats_op!(data, slice, {
        let n = slice.len();
        if n < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "z_scores requires at least two elements",
            ));
        }
        let m = slice.par_iter().sum::<f64>() / n as f64;
        let sd =
            (slice.par_iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        Ok(if sd == 0.0 {
            vec![0.0; n]
        } else {
            slice.par_iter().map(|&x| (x - m) / sd).collect()
        })
    })
}

fn _median_inner(mut data: Vec<f64>) -> PyResult<f64> {
    if data.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "median requires at least one element",
        ));
    }
    if data.iter().any(|x| x.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "median does not support NaN",
        ));
    }
    let len = data.len();
    let mid = len / 2;
    data.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    if len % 2 == 0 {
        let m2 = data[mid];
        let m1 = *data[..mid]
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok((m1 + m2) / 2.0)
    } else {
        Ok(data[mid])
    }
}

#[pyfunction]
pub fn median(data: Bound<'_, PyAny>) -> PyResult<f64> {
    let vec: Vec<f64> = if let Ok(v) = data.extract::<PyRef<Vector>>() {
        v.to_list()
    } else {
        data.extract()?
    };
    _median_inner(vec)
}

#[pyfunction]
pub fn mode(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode requires at least one element",
            ));
        }
        if slice.iter().any(|x| x.is_nan()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode does not support NaN",
            ));
        }
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for &x in slice.iter() {
            *counts.entry(x.to_bits()).or_insert(0) += 1;
        }
        let (&best_bits, _) = counts.iter().max_by_key(|&(_, count)| count).unwrap();
        Ok(f64::from_bits(best_bits))
    })
}

pub fn register_stats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_mean, m)?)?;
    m.add_function(wrap_pyfunction!(harmonic_mean, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(z_scores, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(mode, m)?)?;
    Ok(())
}
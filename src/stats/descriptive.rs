use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vector::Vector;
use std::collections::HashMap;

/// Welford state for parallel reduction (up to 4th moment)
#[derive(Clone, Copy)]
struct WelfordState {
    cnt: f64,
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl WelfordState {
    fn new(x: f64) -> Self {
        Self {
            cnt: 1.0,
            m1: x,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    fn empty() -> Self {
        Self {
            cnt: 0.0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    fn combine(a: Self, b: Self) -> Self {
        if a.cnt == 0.0 { return b; }
        if b.cnt == 0.0 { return a; }

        let n = a.cnt + b.cnt;
        let delta = b.m1 - a.m1;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta3 * delta;

        let m1 = a.m1 + delta * b.cnt / n;
        
        let m2 = a.m2 + b.m2 + delta2 * a.cnt * b.cnt / n;
        
        let m3 = a.m3 + b.m3 
            + delta3 * a.cnt * b.cnt * (a.cnt - b.cnt) / (n * n)
            + 3.0 * delta * (a.cnt * b.m2 - b.cnt * a.m2) / n;
            
        let m4 = a.m4 + b.m4 
            + delta4 * a.cnt * b.cnt * (a.cnt * a.cnt - a.cnt * b.cnt + b.cnt * b.cnt) / (n * n * n)
            + 6.0 * delta2 * (a.cnt * a.cnt * b.m2 + b.cnt * b.cnt * a.m2) / (n * n)
            + 4.0 * delta * (a.cnt * b.m3 - b.cnt * a.m3) / n;

        Self { cnt: n, m1, m2, m3, m4 }
    }
}

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

/// Generate a summary report of descriptive statistics for a dataset.
///
/// Returns a dictionary containing count, mean, variance, standard deviation,
/// skewness, and kurtosis. Uses a parallelized Welford's algorithm for
/// high precision and speed.
///
/// Examples:
///     >>> from rmath.stats import describe
///     >>> stats = describe([1, 2, 3, 4, 5])
///     >>> stats['mean']
///     3.0
#[pyfunction]
pub fn describe(data: Bound<'_, PyAny>) -> PyResult<HashMap<String, f64>> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Cannot describe empty dataset"));
        }

        let state = slice.par_iter()
            .map(|&x| WelfordState::new(x))
            .reduce(|| WelfordState::empty(), WelfordState::combine);

        let mut res = HashMap::new();
        res.insert("count".to_string(), state.cnt);
        res.insert("mean".to_string(), state.m1);
        
        let var = if state.cnt > 1.0 { state.m2 / (state.cnt - 1.0) } else { 0.0 };
        res.insert("variance".to_string(), var);
        res.insert("std".to_string(), var.sqrt());

        if var > 0.0 {
            let s = var.sqrt();
            let skew = (state.cnt.sqrt() * state.m3) / (s * s * s);
            let kurt = (state.cnt * state.m4) / (state.m2 * state.m2) - 3.0;
            res.insert("skewness".to_string(), skew);
            res.insert("kurtosis".to_string(), kurt);
        } else {
            res.insert("skewness".to_string(), 0.0);
            res.insert("kurtosis".to_string(), 0.0);
        }

        Ok(res)
    })
}

/// Calculate the arithmetic mean of a dataset.
///
/// Multi-threaded calculation on the Rayon thread pool. Releases the 
/// Python GIL during execution.
///
/// Examples:
///     >>> from rmath.stats import mean
///     >>> mean([1, 2, 3, 4, 5])
///     3.0
#[pyfunction]
pub fn mean(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() { return Ok(f64::NAN); }
        Ok(slice.par_iter().sum::<f64>() / slice.len() as f64)
    })
}

/// Calculate the sample variance (degree of freedom = 1).
///
/// Uses Welford's algorithm for numerical stability.
#[pyfunction]
pub fn variance(data: Bound<'_, PyAny>) -> PyResult<f64> {
    let d = describe(data)?;
    Ok(*d.get("variance").unwrap_or(&0.0))
}

/// Calculate the sample standard deviation.
///
/// Uses Welford's algorithm for numerical stability.
#[pyfunction]
pub fn std_dev(data: Bound<'_, PyAny>) -> PyResult<f64> {
    let d = describe(data)?;
    Ok(*d.get("std").unwrap_or(&0.0))
}

/// Calculate the median of a dataset.
///
/// Uses an unstable selection algorithm (introselect) for O(N) performance, 
/// avoiding the O(N log N) cost of a full sort.
///
/// Examples:
///     >>> from rmath.stats import median
///     >>> median([3, 1, 2])
///     2.0
///     >>> median([1, 2, 3, 4])  # Average of middle two
///     2.5
#[pyfunction]
pub fn median(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() { return Ok(f64::NAN); }
        let mut v = slice.to_vec();
        let mid = v.len() / 2;
        v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
        if v.len() % 2 == 0 {
            let m1 = v[mid];
            let m0 = *v[..mid].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            Ok((m0 + m1) / 2.0)
        } else {
            Ok(v[mid])
        }
    })
}

/// Find the mode (most common value) of a dataset.
///
/// If there are multiple modes with the same highest frequency, the one 
/// encountered first in the input is returned.
///
/// Examples:
///     >>> from rmath.stats import mode
///     >>> mode([1, 2, 2, 3, 3, 3])
///     3.0
#[pyfunction]
pub fn mode(data: Bound<'_, PyAny>) -> PyResult<f64> {
    dispatch_stats_op!(data, slice, {
        if slice.is_empty() { return Ok(f64::NAN); }
        
        // (count, first_index)
        let mut counts: HashMap<u64, (usize, usize)> = HashMap::new();
        for (i, &x) in slice.iter().enumerate() {
            let entry = counts.entry(x.to_bits()).or_insert((0, i));
            entry.0 += 1;
        }

        // Find max frequency, then stable first index
        let (&bits, _) = counts.iter().max_by(|a, b| {
            let (count_a, first_a) = a.1;
            let (count_b, first_b) = b.1;
            if count_a != count_b {
                count_a.cmp(count_b)
            } else {
                // For ties, smaller index (first encounter) wins
                first_b.cmp(first_a)
            }
        }).unwrap();
        
        Ok(f64::from_bits(bits))
    })
}

/// Calculate the Interquartile Range (IQR).
///
/// Measures the difference between the 75th and 25th percentiles.
/// Highly robust to outliers.
///
/// Examples:
///     >>> from rmath.stats import iqr
///     >>> iqr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
///     4.5
#[pyfunction]
pub fn iqr(data: Bound<'_, PyAny>) -> PyResult<f64> {
    let v: PyRef<Vector> = data.extract()?;
    if v.is_empty() { return Ok(f64::NAN); }
    
    // Custom sort-once logic for IQR
    let mut sorted = v.with_slice(|s| s.to_vec());
    sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len() as f64;
    let get_p = |p: f64| {
        let idx = p / 100.0 * (n - 1.0);
        let i = idx.floor() as usize;
        let fract = idx.fract();
        if i + 1 < sorted.len() {
            sorted[i] + fract * (sorted[i+1] - sorted[i])
        } else {
            sorted[i]
        }
    };
    
    Ok(get_p(75.0) - get_p(25.0))
}

/// Calculate the Median Absolute Deviation (MAD).
///
/// A robust measure of the variability of a univariate sample of 
/// quantitative data.
#[pyfunction]
pub fn mad(data: Bound<'_, PyAny>) -> PyResult<f64> {
    let v: PyRef<Vector> = data.extract()?;
    if v.is_empty() { return Ok(f64::NAN); }
    let m = v.median();
    let abs_dev = v.sub_scalar(m).abs();
    Ok(abs_dev.median())
}

/// Calculate the excess kurtosis of a dataset.
#[pyfunction]
pub fn kurtosis(data: Bound<'_, PyAny>) -> PyResult<f64> {
    let d = describe(data)?;
    Ok(*d.get("kurtosis").unwrap_or(&0.0))
}

/// Calculate the skewness of a dataset.
#[pyfunction]
pub fn skewness(data: Bound<'_, PyAny>) -> PyResult<f64> {
    let d = describe(data)?;
    Ok(*d.get("skewness").unwrap_or(&0.0))
}

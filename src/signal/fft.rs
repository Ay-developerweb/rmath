use pyo3::prelude::*;
use crate::vector::{Vector, complex::ComplexVector};
use num_complex::Complex64;
use rustfft::{FftPlanner, Fft};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use once_cell::sync::Lazy;

type FftCache = Mutex<HashMap<(usize, bool), Arc<dyn Fft<f64>>>>;
static PLANNER: Lazy<Mutex<FftPlanner<f64>>> = Lazy::new(|| Mutex::new(FftPlanner::new()));
static CACHE: Lazy<FftCache> = Lazy::new(|| Mutex::new(HashMap::new()));

fn get_fft(n: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    let mut cache = CACHE.lock().unwrap();
    if let Some(fft) = cache.get(&(n, inverse)) {
        return Arc::clone(fft);
    }
    let mut planner = PLANNER.lock().unwrap();
    let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
    cache.insert((n, inverse), Arc::clone(&fft));
    fft
}

/// Internal FFT kernel – zero FFI overhead
pub fn fft_internal(data: &[f64], n_fft: usize) -> Vec<Complex64> {
    let mut buffer = vec![Complex64::new(0.0, 0.0); n_fft];
    for (i, &x) in data.iter().enumerate() {
        buffer[i] = Complex64::new(x, 0.0);
    }
    let fft = get_fft(n_fft, false);
    fft.process(&mut buffer);
    buffer
}

/// Internal IFFT kernel – zero FFI overhead
pub fn ifft_internal(mut data: Vec<Complex64>) -> Vec<f64> {
    let n = data.len();
    let fft = get_fft(n, true);
    fft.process(&mut data);
    data.iter().map(|c| c.re / n as f64).collect()
}

/// Compute the one-dimensional Fast Fourier Transform (FFT).
///
/// Returns a `ComplexVector` containing the frequency domain representation.
/// Uses a cached global planner for maximum performance.
///
/// Examples:
///     >>> from rmath.vector import Vector
///     >>> from rmath.signal import fft
///     >>> sig = Vector([1, 0, 1, 0])
///     >>> res = fft(sig)
///     >>> len(res)
///     4
#[pyfunction]
pub fn fft(data_any: Bound<'_, PyAny>) -> PyResult<ComplexVector> {
    let v = if let Ok(v_ref) = data_any.extract::<PyRef<Vector>>() { v_ref.clone() }
    else { let list: Vec<f64> = data_any.extract()?; Vector::new(list) };

    let n = v.len_internal();
    if n == 0 { return Err(pyo3::exceptions::PyValueError::new_err("Empty signal")); }

    let buffer = v.with_slice(|slice| fft_internal(slice, n));
    Ok(ComplexVector { data: buffer })
}

/// Compute the inverse Fast Fourier Transform (IFFT).
///
/// Returns a real-valued `Vector` representing the time-domain signal.
///
/// Examples:
///     >>> from rmath.signal import fft, ifft
///     >>> sig = [1.0, 0.0, 1.0, 0.0]
///     >>> ifft(fft(sig))
///     Vector([1.0, 0.0, 1.0, 0.0])
#[pyfunction]
pub fn ifft(cv: &ComplexVector) -> PyResult<Vector> {
    let res = ifft_internal(cv.data.clone());
    Ok(Vector::new(res))
}

/// Compute the FFT and return both magnitude and phase components.
///
/// Returns a tuple of (Magnitudes, Phases) as real-valued Vectors.
///
/// Examples:
///     >>> from rmath.signal import fft_styled
///     >>> mags, phases = fft_styled([1, 0, 1, 0])
#[pyfunction]
pub fn fft_styled(data_any: Bound<'_, PyAny>) -> PyResult<(Vector, Vector)> {
    let cv = fft(data_any)?;
    Ok((cv.to_mags(), cv.to_phases()))
}

/// Real-input FFT – returns ONLY magnitudes for the positive frequencies.
///
/// For a signal of length N, returns magnitudes for the first floor(N/2) + 1 bins.
/// This is the most common use case for power spectrum analysis.
///
/// Examples:
///     >>> from rmath.signal import rfft
///     >>> spectrum = rfft([1, 0, 1, 0, 1, 0, 1, 0])
///     >>> len(spectrum)
///     5
#[pyfunction]
pub fn rfft(data_any: Bound<'_, PyAny>) -> PyResult<Vector> {
    let cv = fft(data_any)?;
    let mags = cv.to_mags();
    let n = mags.len_internal();
    let half_n = (n / 2) + 1;
    Ok(mags.head(half_n))
}

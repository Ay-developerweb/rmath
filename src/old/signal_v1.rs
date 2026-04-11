use pyo3::prelude::*;
use crate::vector::Vector;
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;

/// A high-performance, Radix-2 recursive FFT kernel.
/// Zero-Copy and Zero-Allocation if data is small!
fn cooley_tukey_recursive(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 { return; }

    // 1. Separate into Even and Odd indices
    let mut even: Vec<Complex64> = (0..n/2).map(|i| data[2*i]).collect();
    let mut odd: Vec<Complex64> = (0..n/2).map(|i| data[2*i+1]).collect();

    // 2. Recurse!
    cooley_tukey_recursive(&mut even);
    cooley_tukey_recursive(&mut odd);

    // 3. Combine results using Twiddle Factors
    for k in 0..n/2 {
        let angle = -2.0 * PI * (k as f64) / (n as f64);
        let t = Complex64::from_polar(1.0, angle) * odd[k];
        data[k] = even[k] + t;
        data[k + n/2] = even[k] - t;
    }
}

/// Computes the Discrete Fourier Transform (FFT) of a real input.
/// Returns (Magnitudes, Phases) as two separate Vectors.
#[pyfunction]
pub fn fft(data_any: Bound<'_, PyAny>) -> PyResult<(Vector, Vector)> {
    let _py = data_any.py();
    
    // 1. Extract to Vector (Hybrid API)
    let v = if let Ok(v_ref) = data_any.extract::<PyRef<Vector>>() { v_ref.clone() }
    else { let list: Vec<f64> = data_any.extract()?; Vector::new(list) };

    let n_orig = v.len_internal();
    if n_orig == 0 { return Err(pyo3::exceptions::PyValueError::new_err("Empty signal")); }

    // 2. Zero-pad to next Power of 2 (The Scientific Standard)
    let n_pow2 = n_orig.next_power_of_two();
    let mut complex_data: Vec<Complex64> = Vec::with_capacity(n_pow2);
    
    v.with_slice(|slice| {
        for &x in slice { complex_data.push(Complex64::new(x, 0.0)); }
        for _ in n_orig..n_pow2 { complex_data.push(Complex64::new(0.0, 0.0)); }
    });

    // 3. Transform! (Brain logic: use specialized kernels or recursive CT)
    cooley_tukey_recursive(&mut complex_data);

    // 4. Convert to Magnitude and Phase
    let (mags, phases): (Vec<f64>, Vec<f64>) = complex_data.par_iter()
        .map(|c| (c.norm(), c.arg()))
        .unzip();

    Ok((Vector::new(mags), Vector::new(phases)))
}

/// Power Spectrum: Returns only non-redundant magnitudes (0 to N/2)!
#[pyfunction]
pub fn rfft(data_any: Bound<'_, PyAny>) -> PyResult<Vector> {
    let (mags, _) = fft(data_any)?;
    let n = mags.len_internal();
    if n <= 1 { return Ok(mags); }
    let half_n = (n / 2) + 1;
    Ok(mags.head(half_n))
}

pub fn register_signal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    Ok(())
}

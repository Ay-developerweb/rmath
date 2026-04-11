use pyo3::prelude::*;
use crate::vector::Vector;
// use num_complex::Complex64;
use crate::signal::fft::{fft_internal, ifft_internal};
// use crate::vector::complex::ComplexVector;

/// Perform 1D convolution of two vectors using the FFT method.
///
/// This implementation uses a Fast Fourier Transform to achieve O(N log N)
/// complexity, making it significantly faster than direct convolution for large inputs.
///
/// Args:
///     signal: The input signal vector.
///     kernel: The filter kernel vector.
///     mode: The convolution mode:
///         - 'full': Return the full linear convolution (length N + M - 1).
///         - 'same': Return output of same length as 'signal'.
///         - 'valid': Return only those elements that do not rely on zero-padding.
///
/// Examples:
///     >>> from rmath.vector import Vector
///     >>> from rmath.signal import convolve
///     >>> s = Vector([1, 2, 3])
///     >>> k = Vector([0, 1, 0])
///     >>> convolve(s, k, 'full')
///     Vector([0, 1, 2, 3, 0])
#[pyfunction]
pub fn convolve(_py: Python<'_>, signal: &Vector, kernel: &Vector, mode: &str) -> PyResult<Vector> {
    signal.with_slice(|s| kernel.with_slice(|k| {
        let n = s.len();
        let m = k.len();
        if n == 0 || m == 0 { return Ok(Vector::new(vec![])); }

        // Use FFT Convolution: O(N log N)
        // 1. Pad to next power of 2
        let out_len = n + m - 1;
        let n_fft = out_len.next_power_of_two();
        
        // FFT(signal) - Direct Rust call
        let s_fft = fft_internal(s, n_fft);

        // FFT(kernel) - Direct Rust call
        let k_fft = fft_internal(k, n_fft);

        // Pointwise Multiply
        let mut res_fft_data = Vec::with_capacity(n_fft);
        for i in 0..n_fft {
            res_fft_data.push(s_fft[i] * k_fft[i]);
        }

        // IFFT - Direct Rust call
        let mut full_res = ifft_internal(res_fft_data);
        
        // Truncate to out_len (n + m - 1)
        full_res.truncate(out_len);

        match mode {
            "full" => Ok(Vector::new(full_res)),
            "same" => {
                let start = (m - 1) / 2;
                Ok(Vector::new(full_res.drain(start..start+n).collect()))
            },
            "valid" => {
                if n < m { return Ok(Vector::new(vec![])); }
                let start = m - 1;
                let len = n - m + 1;
                Ok(Vector::new(full_res.drain(start..start+len).collect()))
            },
            _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid mode")),
        }
    }))
}

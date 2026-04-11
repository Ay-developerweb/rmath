use pyo3::prelude::*;

pub mod fft;
pub mod convolution;

pub fn register_signal(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft::fft, m)?)?;
    m.add_function(wrap_pyfunction!(fft::fft_styled, m)?)?;
    m.add_function(wrap_pyfunction!(fft::rfft, m)?)?;
    m.add_function(wrap_pyfunction!(fft::ifft, m)?)?;
    m.add_function(wrap_pyfunction!(convolution::convolve, m)?)?;
    Ok(())
}

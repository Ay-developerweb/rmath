use pyo3::prelude::*;

/// A high-performance, Rust-backed scalar value.
/// 
/// Scalar objects wrap a raw 64-bit float and provide native-speed
/// arithmetic by performing computations directly in Rust.
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scalar(pub f64);

#[pymethods]
impl Scalar {
    #[new]
    pub fn new(value: f64) -> Self {
        Scalar(value)
    }

    /// Returns the underlying Python float value.
    pub fn to_python(&self) -> f64 {
        self.0
    }

    pub fn __repr__(&self) -> String {
        format!("Scalar({})", self.0)
    }

    pub fn __str__(&self) -> String {
        self.0.to_string()
    }

    pub fn __float__(&self) -> f64 {
        self.0
    }
}

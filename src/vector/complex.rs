use pyo3::prelude::*;
use crate::vector::Vector;
use num_complex::Complex64;
use rayon::prelude::*;

/// A specialized container for complex-valued (f64 + i*f64) vectors.
///
/// `ComplexVector` provides optimized operations for Fourier-domain or
/// analytical signal data where complex numbers are required.
#[pyclass(module = "rmath")]
#[derive(Clone)]
pub struct ComplexVector {
    pub data: Vec<Complex64>,
}

#[pymethods]
impl ComplexVector {
    /// Create a new ComplexVector from real and imaginary components.
    ///
    /// Examples:
    ///     >>> from rmath.vector import ComplexVector
    ///     >>> cv = ComplexVector([1, 2], [0, 1])
    ///     >>> cv
    ///     ComplexVector([1+0j, 2+1j])
    #[new]
    pub fn new(real: Vec<f64>, imag: Vec<f64>) -> PyResult<Self> {
        if real.len() != imag.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch"));
        }
        let data = real.into_iter().zip(imag.into_iter())
            .map(|(r, i)| Complex64::new(r, i))
            .collect();
        Ok(ComplexVector { data })
    }

    /// Return a Vector of the magnitudes (L2 norms) of each complex element.
    pub fn to_mags(&self) -> Vector {
        let mags = self.data.par_iter().map(|c| c.norm()).collect();
        Vector::new(mags)
    }

    /// Return a Vector of the phases (angles in radians) of each complex element.
    pub fn to_phases(&self) -> Vector {
        let phases = self.data.par_iter().map(|c| c.arg()).collect();
        Vector::new(phases)
    }

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Apply a boolean mask in-place, zeroing out elements where the mask is false.
    /// Highly efficient for frequency filtering in DSP.
    pub fn mask_inplace(&mut self, mask: Vec<bool>) -> PyResult<()> {
        if mask.len() != self.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Mask length mismatch"));
        }
        for (i, &keep) in mask.iter().enumerate() {
            if !keep {
                self.data[i] = Complex64::new(0.0, 0.0);
            }
        }
        Ok(())
    }
}

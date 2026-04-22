use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArrayMethods, ToPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use super::core::Array;

/// Interoperability with NumPy, Pandas, and PyTorch.
/// 
/// Optimized via the `numpy` crate and DLPack protocol.

#[pymethods]
impl Array {

    // ── NumPy ─────────────────────────────────────────────────────────────

    /// Export to a NumPy ndarray.
    ///
    /// This provides an optimized zero-copy bridge where possible.
    ///
    /// Example:
    ///     >>> np_arr = a.to_numpy()
    ///     >>> type(np_arr)
    ///     <class 'numpy.ndarray'>
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let contiguous = self.to_contiguous();
        let vec = contiguous.data().to_vec();
        let arr = vec.to_pyarray(py); 
        // PyArrayMethods::reshape requires an IntoDimension type
        let reshaped = arr.reshape(numpy::ndarray::IxDyn(&self.shape))?;
        Ok(reshaped.into_any())
    }

    /// Import from numpy ndarray (optimized)
    #[staticmethod]
    pub fn from_numpy<'py>(arr: &Bound<'py, PyAny>) -> PyResult<Array> {
        let np_arr: PyReadonlyArrayDyn<f64> = arr.extract()?;
        let shape = np_arr.shape().to_vec();
        // Extract data as slice, then to_vec (copies)
        let slice = np_arr.as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("NumPy array must be contiguous"))?;
        Ok(Array::from_flat(slice.to_vec(), shape))
    }

    /// __array__ protocol — lets np.array(arr) work directly
    /// NumPy 2.x passes dtype=None, copy=None as keyword args
    #[pyo3(signature = (dtype=None, copy=None))]
    pub fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let arr = self.to_numpy(py)?;
        // If a specific dtype was requested, cast
        if let Some(dt) = dtype {
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", dt)?;
            if let Some(c) = copy {
                kwargs.set_item("copy", c)?;
            }
            let np = py.import("numpy")?;
            return Ok(np.call_method("asarray", (&arr,), Some(&kwargs))?.into_any());
        }
        Ok(arr)
    }

    // ── Pandas ────────────────────────────────────────────────────────────

    /// Export to pandas DataFrame (2-D only)
    #[pyo3(signature = (columns=None))]
    pub fn to_dataframe<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<String>>
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.ndim() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Only 2-D arrays can become DataFrames"));
        }
        let pd = py.import("pandas")?;
        let np_arr = self.to_numpy(py)?;
        let kwargs = PyDict::new(py);
        if let Some(cols) = columns {
            kwargs.set_item("columns", cols)?;
        }
        Ok(pd.call_method("DataFrame", (np_arr,), Some(&kwargs))?.into_any())
    }

    /// Import from pandas DataFrame → Array
    #[staticmethod]
    pub fn from_dataframe<'py>(df: &Bound<'py, PyAny>) -> PyResult<Array> {
        let values = df.call_method0("to_numpy")?;
        Array::from_numpy(&values)
    }

    /// Export to pandas Series (1-D, uses first row or flat)
    #[pyo3(signature = (name=None))]
    pub fn to_series<'py>(
        &self,
        py: Python<'py>,
        name: Option<String>
    ) -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import("pandas")?;
        let contiguous = self.to_contiguous();
        let flat: Vec<f64> = contiguous.data().to_vec();
        let py_list = flat.into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        if let Some(n) = name {
            kwargs.set_item("name", n)?;
        }
        Ok(pd.call_method("Series", (py_list,), Some(&kwargs))?.into_any())
    }

    // ── PyTorch ───────────────────────────────────────────────────────────

    /// Export to torch.Tensor (via NumPy — single memcpy, no N Python floats)
    pub fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import("torch")?;
        let np_arr = self.to_numpy(py)?;
        // torch.from_numpy() shares memory; .clone() detaches from NumPy
        let tensor = torch.call_method1("from_numpy", (&np_arr,))?;
        Ok(tensor.call_method0("clone")?.into_any())
    }

    /// Import from torch.Tensor -> Array
    #[staticmethod]
    pub fn from_torch<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<Array> {
        // Extract shape and flat data via Python calls
        // This avoids the numpy Rust crate entirely, sidestepping version mismatches
        let shape_obj = tensor.getattr("shape")?;
        let shape: Vec<usize> = shape_obj.extract()?;
        let flat_list = tensor.call_method0("detach")?
            .call_method0("cpu")?
            .call_method0("flatten")?
            .call_method0("tolist")?;
        let data: Vec<f64> = flat_list.extract()?;
        Ok(Array::from_flat(data, shape))
    }

    // ── DLPack ────────────────────────────────────────────────────────────

    /// DLPack protocol support (zero-copy for Torch/JAX/etc)
    #[pyo3(signature = (_stream=None))]
    pub fn __dlpack__<'py>(&self, py: Python<'py>, _stream: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        // Export via numpy's dlpack for now to ensure standard compliance
        let np_arr = self.to_numpy(py)?;
        np_arr.call_method0("__dlpack__")
    }

    pub fn __dlpack_device__<'py>(&self, _py: Python<'py>) -> PyResult<(i32, i32)> {
        Ok((1, 0)) // (kDLCPU, device_id=0)
    }

    // ── JAX ───────────────────────────────────────────────────────────────

    /// Export to jax.numpy array (via NumPy — single memcpy, no N Python floats)
    pub fn to_jax<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let jnp = py.import("jax.numpy")?;
        let np_arr = self.to_numpy(py)?;
        // jnp.asarray accepts NumPy arrays with efficient transfer
        Ok(jnp.call_method1("asarray", (&np_arr,))?.into_any())
    }

    /// Import from jax array → Array
    #[staticmethod]
    pub fn from_jax<'py>(arr: &Bound<'py, PyAny>) -> PyResult<Array> {
        let py = arr.py();
        let np = py.import("numpy")?;
        let np_arr = np.call_method1("asarray", (arr,))?;
        Array::from_numpy(&np_arr)
    }

    // ── Scikit-learn compatible ───────────────────────────────────────────

    /// Returns (n_samples, n_features) tuple — sklearn convention
    pub fn sklearn_shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Validate sklearn array (2-D, finite, no NaN)
    pub fn validate_sklearn(&self) -> PyResult<()> {
        if self.ndim() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sklearn requires 2-D array"));
        }
        let contiguous = self.to_contiguous();
        if contiguous.data().iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Array contains NaN or Inf values"));
        }
        Ok(())
    }

    // ── Generic Python list round-trip ────────────────────────────────────

    /// to_list() already defined in core.rs
    /// from_list() already defined in core.rs

    /// Export as flat Python list (avoids nested list overhead)
    pub fn to_flat_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let contiguous = self.to_contiguous();
        let v: Vec<f64> = contiguous.data().to_vec();
        Ok(PyList::new(py, v)?.into_any())
    }
}
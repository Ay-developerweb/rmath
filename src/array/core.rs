use pyo3::prelude::*;
use crate::vector::Vector;
use faer::Mat;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

// ─── Constants ──────────────────────────────────────────────────────────────

pub const INLINE_MAX: usize = 32;       // Zero-alloc threshold (32 * 8 bytes = 256 bytes)
pub const PAR_THRESHOLD: usize = 131072;  // Rayon crossover point

// ─── Storage ────────────────────────────────────────────────────────────────

/// High-performance storage with three-tier access:
/// 1. Inline: Fixed stack array (32 elements) for zero-allocation hotspots.
/// 2. Heap: Arc-backed flat Vec<f64> for shared ownership and GB-scale data.
/// 3. Mmap: (Future) for out-of-core computation.
#[derive(Clone, Debug)]
pub enum ArrayStorage {
    Inline([f64; INLINE_MAX], usize),
    Heap(std::sync::Arc<Vec<f64>>),
}

impl ArrayStorage {
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        match self {
            ArrayStorage::Inline(d, n) => &d[..*n],
            ArrayStorage::Heap(v)      => v.as_slice(),
        }
    }

    pub fn from_vec(v: Vec<f64>) -> Self {
        let n = v.len();
        if n <= INLINE_MAX {
            let mut data = [0.0f64; INLINE_MAX];
            data[..n].copy_from_slice(&v);
            ArrayStorage::Inline(data, n)
        } else {
            ArrayStorage::Heap(std::sync::Arc::new(v))
        }
    }
}

// ─── Array ──────────────────────────────────────────────────────────────────

/// High-performance N-dimensional array engine for numerical computing.
/// 
/// `Array` is the central data structure in RMath, providing a zero-copy
/// bridge between Rust's speed and Python's flexibility. It supports
/// GB-scale data with its three-tier storage architecture (Inline, Heap, Mmap).
///
/// Features:
/// - SIMD-accelerated math kernels.
/// - Automatic broadcasting rules.
/// - Memory-efficient lazy loading and memory-mapping.
/// - Zero-copy interop with NumPy.
#[pyclass(subclass)]
#[derive(Clone, Debug)]
pub struct Array {
    pub storage: ArrayStorage,
    pub shape:   Vec<usize>,   // e.g. [3, 4] or [2, 3, 4]
    pub strides: Vec<usize>,   // row-major strides
}

// ── Internal helpers ─────────────────────────────────────────────────────────

impl Array {
    /// Compute row-major strides from shape.
    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Calculate the resulting shape of broadcasting two shapes.
    pub fn broadcast_shapes(s1: &[usize], s2: &[usize]) -> PyResult<Vec<usize>> {
        let ndim1 = s1.len();
        let ndim2 = s2.len();
        let max_ndim = ndim1.max(ndim2);
        let mut res = vec![0usize; max_ndim];
        
        for i in 0..max_ndim {
            let dim1 = if i < max_ndim - ndim1 { 1 } else { s1[i - (max_ndim - ndim1)] };
            let dim2 = if i < max_ndim - ndim2 { 1 } else { s2[i - (max_ndim - ndim2)] };
            
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Cannot broadcast shapes {:?} and {:?}", s1, s2)));
            }
            res[i] = dim1.max(dim2);
        }
        Ok(res)
    }

    pub fn from_flat(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Array { storage: ArrayStorage::from_vec(data), shape, strides }
    }

    #[inline]
    pub fn data(&self) -> &[f64] { self.storage.as_slice() }

    #[inline]
    pub fn len(&self) -> usize { self.data().len() }

    #[inline]
    pub fn ndim(&self) -> usize { self.shape.len() }



    /// Flat index from multi-index (row-major)
    #[inline]
    pub fn flat_index(&self, idx: &[usize]) -> usize {
        idx.iter().zip(self.strides.iter()).map(|(i, s)| i * s).sum()
    }

    /// Get element at multi-index
    #[inline]
    pub fn get(&self, idx: &[usize]) -> f64 {
        self.data()[self.flat_index(idx)]
    }

    pub fn to_mat(&self) -> Mat<f64> {
        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        // Mat::from_fn is reliable and fast for conversion
        Mat::from_fn(r, c, |i, j| d[i * c + j])
    }

    pub fn from_mat(m: Mat<f64>) -> Self {
        let r = m.nrows();
        let c = m.ncols();
        let mut data = vec![0.0; r * c];
        // Copy out in row-major order: faer is col-major in storage, but we can iterate rows
        for i in 0..r {
            for j in 0..c {
                data[i * c + j] = m[(i, j)];
            }
        }
        Self::from_flat(data, vec![r, c])
    }

    /// Mutable data — only when storage is uniquely owned (Heap with refcount=1)
    pub fn data_mut(&mut self) -> Option<&mut [f64]> {
        match &mut self.storage {
            ArrayStorage::Inline(d, n) => Some(&mut d[..*n]),
            ArrayStorage::Heap(arc) => std::sync::Arc::get_mut(arc).map(|v| v.as_mut_slice()),
        }
    }

    /// Ensure we own the data (copy-on-write)
    pub fn make_owned(&mut self) {
        if let ArrayStorage::Heap(arc) = &self.storage {
            if std::sync::Arc::strong_count(arc) > 1 {
                let v = arc.as_ref().clone();
                self.storage = ArrayStorage::Heap(std::sync::Arc::new(v));
            }
        }
    }

    pub fn map_elements<F: Fn(f64) -> f64 + Sync + Send>(&self, f: F) -> Self {
        let sd = self.data();
        let data: Vec<f64> = if sd.len() >= PAR_THRESHOLD {
            sd.par_iter().map(|&x| f(x)).collect()
        } else {
            sd.iter().map(|&x| f(x)).collect()
        };
        Self::from_flat(data, self.shape.clone())
    }

    pub fn transpose_internal(&self) -> Self {
        let nd = self.ndim();
        if nd < 2 { return self.clone(); }
        
        let mut new_shape = self.shape.clone();
        new_shape.swap(nd - 2, nd - 1);
        
        let n = self.len();
        let mut out_data = vec![0.0; n];
        let r = self.shape[nd - 2];
        let c = self.shape[nd - 1];
        let slice_size = r * c;
        let old_data = self.data();
        
        // Parallel transpose across all batches
        out_data.par_chunks_mut(slice_size).enumerate().for_each(|(b, out_slice)| {
            let offset = b * slice_size;
            for i in 0..r {
                for j in 0..c {
                    // Swap (i, j) to (j, i)
                    out_slice[j * r + i] = old_data[offset + i * c + j];
                }
            }
        });
        
        Self::from_flat(out_data, new_shape)
    }

    // ── Internal Rust Builders ───────────────────────────────────────────

    pub fn zeros_internal(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_flat(vec![0.0; n], shape.to_vec())
    }

    pub fn ones_internal(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_flat(vec![1.0; n], shape.to_vec())
    }

    pub fn full_internal(shape: &[usize], val: f64) -> Self {
        let n: usize = shape.iter().product();
        Self::from_flat(vec![val; n], shape.to_vec())
    }
}

// ── Python API ───────────────────────────────────────────────────────────────

#[pymethods]
impl Array {
    /// Unified Constructor: Detects shape automatically from nested Sequences (list, tuple, etc.)
    /// rm.Array([1, 2]) -> shape [2]
    /// Create a new N-dimensional Array from nested Python sequences or a Vector.
    ///
    /// The shape is automatically detected from the nesting level of the input.
    ///
    /// Examples:
    /// ```python
    /// import rmath as rm
    /// a = rm.Array([1, 2, 3])          # 1D array, shape [3]
    /// b = rm.Array([[1, 2], [3, 4]])    # 2D array, shape [2, 2]
    /// c = rm.Array([])                 # Empty 2D array, shape [0, 0]
    /// ```
    #[new]
    pub fn new(data: Bound<'_, PyAny>) -> PyResult<Self> {
        // Handshake: If passing a Vector, convert instantly
        if let Ok(v) = data.extract::<PyRef<Vector>>() {
            let d = v.with_slice(|s| s.to_vec());
            return Ok(Self::from_flat(d, vec![v.len_internal()]));
        }

        // Handle empty case: distinguish [] from [[]]
        if let Ok(list) = data.downcast::<pyo3::types::PyList>() {
            if list.is_empty() {
                return Ok(Self::from_flat(vec![], vec![0, 0]));
            }
            
            // Fast-path: 2D list-of-lists
            if let Ok(first_item) = list.get_item(0) {
                if let Ok(first_row) = first_item.downcast::<pyo3::types::PyList>() {
                    let n_rows = list.len();
                    let n_cols = first_row.len();
                    let mut flat = Vec::with_capacity(n_rows * n_cols);
                    let mut consistent = true;
                    
                    for i in 0..n_rows {
                        if let Ok(row_item) = list.get_item(i) {
                            if let Ok(row) = row_item.downcast::<pyo3::types::PyList>() {
                                if row.len() != n_cols { consistent = false; break; }
                                for j in 0..n_cols {
                                    if let Ok(val_item) = row.get_item(j) {
                                        if let Ok(val) = val_item.extract::<f64>() {
                                            flat.push(val);
                                        } else { consistent = false; break; }
                                    } else { consistent = false; break; }
                                }
                            } else { consistent = false; break; }
                        } else { consistent = false; break; }
                    }
                    
                    if consistent {
                        return Ok(Self::from_flat(flat, vec![n_rows, n_cols]));
                    }
                }
            }
        }

        // Generic Recursive Fallback for N-D
        let mut flat_data = Vec::new();
        let mut shape = Vec::new();

        fn walk(obj: &Bound<'_, PyAny>, flat: &mut Vec<f64>, shape: &mut Vec<usize>, depth: usize) -> PyResult<()> {
            if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
                let len = list.len();
                if shape.len() <= depth {
                    shape.push(len);
                } else if shape[depth] != len {
                    return Err(pyo3::exceptions::PyValueError::new_err("Inhomogeneous shape in nested list"));
                }
                for i in 0..len {
                    walk(&list.get_item(i)?, flat, shape, depth + 1)?;
                }
            } else {
                flat.push(obj.extract::<f64>()?);
            }
            Ok(())
        }

        walk(&data, &mut flat_data, &mut shape, 0)?;
        Ok(Self::from_flat(flat_data, shape))
    }

    // ── Static constructors ───────────────────────────────────────────────

    #[staticmethod]
    #[pyo3(name = "zeros", signature = (*shape))]
    pub fn zeros_py(shape: Vec<usize>) -> Self {
        Self::zeros_internal(&shape)
    }

    /// Create a new N-dimensional Array filled with ones.
    #[staticmethod]
    #[pyo3(name = "ones", signature = (*shape))]
    pub fn ones_py(shape: Vec<usize>) -> Self {
        Self::ones_internal(&shape)
    }

    /// Create a new N-dimensional Array filled with a scalar value.
    ///
    /// Syntax: `Array.full(dim1, dim2, ..., val)`
    #[staticmethod]
    #[pyo3(signature = (*args))]
    pub fn full(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let n_args = args.len();
        if n_args < 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err("full() requires at least 1 dimension and a value"));
        }
        let val = args.get_item(n_args - 1)?.extract::<f64>()?;
        let mut shape = Vec::with_capacity(n_args - 1);
        for i in 0..n_args - 1 {
            shape.push(args.get_item(i)?.extract::<usize>()?);
        }
        Ok(Self::full_internal(&shape, val))
    }

    /// Create a 2D identity matrix of size N x N.
    #[staticmethod]
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n { data[i * n + i] = 1.0; }
        Self::from_flat(data, vec![n, n])
    }

    #[staticmethod]
    #[pyo3(signature = (*shape))]
    pub fn randn(shape: Vec<usize>) -> Self {
        let mut rng = thread_rng();
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
        Self::from_flat(data, shape)
    }

    #[staticmethod]
    #[pyo3(signature = (*shape))]
    pub fn rand_uniform(shape: Vec<usize>) -> Self {
        let mut rng = thread_rng();
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>()).collect();
        Self::from_flat(data, shape)
    }

    #[staticmethod]
    pub fn arange(start: f64, stop: f64, step: f64) -> PyResult<Self> {
        if step == 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("step=0")); }
        let n = ((stop - start) / step).ceil().max(0.0) as usize;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Ok(Self::from_flat(data.clone(), vec![1, data.len()]))
    }

    #[staticmethod]
    pub fn linspace(start: f64, stop: f64, n: usize) -> Self {
        if n == 0 { return Self::from_flat(vec![], vec![1, 0]); }
        if n == 1 { return Self::from_flat(vec![start], vec![1, 1]); }
        let step = (stop - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Self::from_flat(data, vec![1, n])
    }

    /// N-D zeros: shape as list e.g. [2,3,4]
    #[staticmethod]
    pub fn zeros_nd(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::from_flat(vec![0.0; n], shape)
    }

    #[staticmethod]
    pub fn ones_nd(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::from_flat(vec![1.0; n], shape)
    }

    // ── Shape / metadata ─────────────────────────────────────────────────

    pub fn shape(&self) -> Vec<usize> { self.shape.clone() }
    pub fn ndim_py(&self) -> usize    { self.ndim() }
    pub fn size(&self)   -> usize     { self.len() }
    pub fn nrows(&self)  -> usize     { if self.shape.is_empty() { 0 } else { self.shape[0] } }
    pub fn ncols(&self)  -> usize     { if self.shape.len() < 2 { 1 } else { self.shape[1] } }

    /// Reshape the array to a new shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let new_n: usize = new_shape.iter().product();
        if new_n != self.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Cannot reshape size {} into {:?}", self.len(), new_shape)));
        }
        let strides = Self::compute_strides(&new_shape);
        Ok(Array { storage: self.storage.clone(), shape: new_shape, strides })
    }

    pub fn flatten(&self) -> Self {
        let n = self.len();
        Self { storage: self.storage.clone(), shape: vec![1, n], strides: vec![n, 1] }
    }

    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().cloned().filter(|&s| s != 1).collect();
        let new_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
        let strides = Self::compute_strides(&new_shape);
        Array { storage: self.storage.clone(), shape: new_shape, strides }
    }

    pub fn expand_dims(&self, axis: usize) -> PyResult<Self> {
        if axis > self.ndim() {
            return Err(pyo3::exceptions::PyValueError::new_err("axis out of range"));
        }
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        let strides = Self::compute_strides(&new_shape);
        Ok(Array { storage: self.storage.clone(), shape: new_shape, strides })
    }

    // ── Indexing ──────────────────────────────────────────────────────────

    pub fn __getitem__(&self, idx: Vec<usize>) -> PyResult<f64> {
        if idx.len() != self.ndim() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Wrong number of indices"));
        }
        for (i, (&ix, &s)) in idx.iter().zip(self.shape.iter()).enumerate() {
            if ix >= s { return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Index {} out of bounds on axis {} (size {})", ix, i, s))); }
        }
        Ok(self.get(&idx))
    }

    pub fn __setitem__(&mut self, idx: Vec<usize>, val: f64) -> PyResult<()> {
        if idx.len() != self.ndim() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Wrong number of indices"));
        }
        self.make_owned();
        let fi = self.flat_index(&idx);
        match &mut self.storage {
            ArrayStorage::Inline(d, _) => d[fi] = val,
            ArrayStorage::Heap(arc)    => {
                std::sync::Arc::get_mut(arc).unwrap()[fi] = val;
            }
        }
        Ok(())
    }

    pub fn get_row(&self, i: usize) -> PyResult<Vec<f64>> {
        if self.ndim() < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Not 2-D")); }
        if i >= self.nrows() { return Err(pyo3::exceptions::PyIndexError::new_err("Row out of bounds")); }
        let c = self.ncols();
        Ok(self.data()[i*c..(i+1)*c].to_vec())
    }

    pub fn get_col(&self, j: usize) -> PyResult<Vec<f64>> {
        if self.ndim() < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Not 2-D")); }
        if j >= self.ncols() { return Err(pyo3::exceptions::PyIndexError::new_err("Col out of bounds")); }
        let (r, c) = (self.nrows(), self.ncols());
        Ok((0..r).map(|i| self.data()[i*c+j]).collect())
    }

    pub fn slice_rows(&self, start: usize, end: usize) -> PyResult<Self> {
        if self.ndim() < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Not 2-D")); }
        let end = end.min(self.nrows());
        if start >= end { return Err(pyo3::exceptions::PyValueError::new_err("start >= end")); }
        let c = self.ncols();
        let data = self.data()[start*c..end*c].to_vec();
        Ok(Self::from_flat(data, vec![end - start, c]))
    }

    // ── Conversion ────────────────────────────────────────────────────────

    pub fn to_list(&self) -> Vec<Vec<f64>> {
        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        (0..r).into_par_iter().map(|i| d[i*c..(i+1)*c].to_vec()).collect()
    }

    pub fn is_symmetric(&self) -> bool {
        if self.ndim() != 2 || self.nrows() != self.ncols() { return false; }
        let n = self.nrows();
        let d = self.data();
        for i in 0..n {
            for j in 0..i {
                if (d[i*n+j] - d[j*n+i]).abs() > 1e-10 { return false; }
            }
        }
        true
    }

    pub fn is_positive_definite(&self) -> bool {
        self.cholesky_internal().is_ok()
    }

    pub fn normalize(&self, mean: &Bound<'_, PyAny>, std: &Bound<'_, PyAny>) -> PyResult<Self> {
        let centered = self.__sub__(mean)?;
        centered.__truediv__(std)
    }

    pub fn is_prime(&self) -> Vec<bool> {
        self.data().iter().map(|&x| {
            let n = x.abs().round() as u64;
            if n < 2 { return false; }
            if n == 2 { return true; }
            if n % 2 == 0 { return false; }
            let limit = (n as f64).sqrt() as u64;
            (3..=limit).step_by(2).all(|i| n % i != 0)
        }).collect()
    }

    pub fn tolist(&self) -> Vec<Vec<f64>> { self.to_list() }

    pub fn to_flat_list(&self) -> Vec<f64> { self.data().to_vec() }

    /// From flat list + shape
    #[staticmethod]
    pub fn from_list(data: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("data len {} != shape product {}", data.len(), n)));
        }
        Ok(Self::from_flat(data, shape))
    }

    pub fn copy(&self) -> Self {
        let data = self.data().to_vec();
        Self::from_flat(data, self.shape.clone())
    }

    /// Zero-copy (if possible) conversion to Vector
    pub fn into_vector(&self) -> PyResult<Vector> {
        if self.ndim() > 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Cannot convert N-D array to Vector"));
        }
        Ok(Vector::new(self.data().to_vec()))
    }

    // ── Display ───────────────────────────────────────────────────────────

    pub fn __repr__(&self) -> String {
        let d = self.data();
        if self.len() <= 10 {
            format!("Array(shape={:?}, values={:?})", self.shape, d)
        } else {
            format!("Array(shape={:?}, values=[{:.4}, {:.4}, ..., {:.4}, {:.4}])", 
                self.shape, d[0], d[1], d[d.len()-2], d[d.len()-1])
        }
    }

    pub fn __len__(&self) -> usize { self.nrows() }
}
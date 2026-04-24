use crate::vector::Vector;
use faer::Mat;
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

// ─── Constants ──────────────────────────────────────────────────────────────

pub const INLINE_MAX: usize = 32; // Zero-alloc threshold (32 * 8 bytes = 256 bytes)
pub const PAR_THRESHOLD: usize = 8_192; // Rayon crossover — unified with Vector

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
            ArrayStorage::Heap(v) => v.as_slice(),
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
/// Examples:
///     >>> import rmath.array as ra
///     >>> a = ra.Array([[1, 2], [3, 4]])
///     >>> a.shape
///     [2, 2]
///     >>> (a * 2).exp()
///     Array([[7.389, 54.598], [403.429, 2980.958]])
#[pyclass(subclass, module = "rmath.array")]
#[derive(Clone, Debug)]
pub struct Array {
    pub storage: ArrayStorage,
    pub shape: Vec<usize>,   // e.g. [3, 4] or [2, 3, 4]
    pub strides: Vec<isize>, // potentially non-contiguous strides
    pub offset: usize,       // data pointer offset for slicing
    #[pyo3(get)]
    pub dtype: String, // primitive choice tag (e.g. "float64", "float32")
}

// ── Internal helpers ─────────────────────────────────────────────────────────

impl Array {
    /// Compute row-major strides from shape.
    pub fn compute_strides(shape: &[usize]) -> Vec<isize> {
        let ndim = shape.len();
        let mut strides = vec![1isize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as isize;
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
            let dim1 = if i < max_ndim - ndim1 {
                1
            } else {
                s1[i - (max_ndim - ndim1)]
            };
            let dim2 = if i < max_ndim - ndim2 {
                1
            } else {
                s2[i - (max_ndim - ndim2)]
            };

            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    s1, s2
                )));
            }
            res[i] = dim1.max(dim2);
        }
        Ok(res)
    }

    pub fn from_flat(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        Array {
            storage: ArrayStorage::from_vec(data),
            shape,
            strides,
            offset: 0,
            dtype: "float64".to_string(),
        }
    }

    #[inline]
    /// Access the underlying raw storage slice.
    /// WARNING: This ignores strides and offset. Use only for contiguous IO or copying.
    pub fn storage_slice(&self) -> &[f64] {
        self.storage.as_slice()
    }

    /// Returns a flat slice of the data if the array is contiguous.
    /// Panics if the array is a non-contiguous view.
    pub fn data(&self) -> &[f64] {
        if !self.is_contiguous() {
            panic!("Attempted to access non-contiguous Array as a flat slice. Call .to_contiguous() first.");
        }
        &self.storage_slice()[self.offset..self.offset + self.len()]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Flat index from multi-index (based on strides and offset)
    #[inline]
    pub fn flat_index(&self, idx: &[usize]) -> usize {
        let mut res = self.offset as isize;
        for (&i, &s) in idx.iter().zip(self.strides.iter()) {
            res += i as isize * s;
        }
        res as usize
    }

    /// Get element at multi-index
    #[inline]
    pub fn get(&self, idx: &[usize]) -> f64 {
        self.storage_slice()[self.flat_index(idx)]
    }

    pub fn to_mat(&self) -> Mat<f64> {
        let (r, c) = (self.nrows(), self.ncols());
        // For efficiency, check if we're dealing with a 2D-friendly view
        if self.ndim() == 2 {
            Mat::from_fn(r, c, |i, j| self.get(&[i, j]))
        } else {
            panic!("to_mat only supported for 2D arrays");
        }
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

    /// In-place addition: self += other (handles broadcasting)
    pub fn add_assign_array(&mut self, other: &Array) -> PyResult<()> {
        if self.shape == other.shape {
            // Fast path: same shape, no broadcasting needed
            self.make_owned();
            let n = self.len();
            let self_data = self.storage_slice_mut();
            let other_data = other.to_contiguous();
            let od = other_data.data();

            if n >= PAR_THRESHOLD {
                self_data[..n]
                    .par_iter_mut()
                    .zip(od.par_iter())
                    .for_each(|(a, &b)| *a += b);
            } else {
                for i in 0..n {
                    self_data[i] += od[i];
                }
            }
            Ok(())
        } else {
            // Slow path: broadcasting needed.
            let res = self.add_array(other)?;
            *self = res;
            Ok(())
        }
    }

    /// Access raw storage mutably. WARNING: Check ownership and strides first.
    pub fn storage_slice_mut(&mut self) -> &mut [f64] {
        self.make_owned();
        match &mut self.storage {
            ArrayStorage::Inline(d, n) => &mut d[..*n],
            ArrayStorage::Heap(arc) => std::sync::Arc::get_mut(arc).unwrap().as_mut_slice(),
        }
    }

    pub fn map_elements<F: Fn(f64) -> f64 + Sync + Send>(&self, f: F) -> Self {
        let contig = self.to_contiguous();
        let sd = contig.storage_slice(); // Safe bypass! We know contig is offset 0 and perfectly ordered
        let data: Vec<f64> = if contig.len() >= PAR_THRESHOLD {
            sd[..contig.len()].par_iter().map(|&x| f(x)).collect()
        } else {
            sd[..contig.len()].iter().map(|&x| f(x)).collect()
        };
        Self::from_flat(data, self.shape.clone())
    }

    /// Transpose the array (Zero-Copy View).
    ///
    /// This is an O(1) operation that simply swaps strides and shape metadata.
    pub fn transpose_internal(&self) -> Self {
        let nd = self.ndim();
        if nd < 2 {
            return self.clone();
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(nd - 2, nd - 1);
        new_strides.swap(nd - 2, nd - 1);

        Array {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype.clone(),
        }
    }

    /// Check if the array is stored contiguously in row-major order.
    pub fn is_contiguous(&self) -> bool {
        let expected = Self::compute_strides(&self.shape);
        self.strides == expected
    }

    /// Ensure the array is contiguous, copying if necessary.
    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }

        let n = self.len();
        let mut data = vec![0.0; n];

        // Fast path: if the array is 2D, we can use a nested loop which is much faster than
        // a general N-D walker.
        if self.ndim() == 2 {
            let r = self.shape[0];
            let c = self.shape[1];
            let rs = self.strides[0];
            let cs = self.strides[1];
            let slice = self.storage_slice();
            let offset = self.offset;

            for i in 0..r {
                let row_offset = offset as isize + i as isize * rs;
                for j in 0..c {
                    data[i * c + j] = slice[(row_offset + j as isize * cs) as usize];
                }
            }
        } else {
            // General N-D fallback (still better than the recursive version by using a flat buffer for indices)
            let mut idx = vec![0; self.ndim()];
            for i in 0..n {
                data[i] = self.get(&idx);
                // Advance multi-index
                for d in (0..self.ndim()).rev() {
                    idx[d] += 1;
                    if idx[d] < self.shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }

        Self::from_flat(data, self.shape.clone())
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

    // Internal helper for dtype routing
    pub fn apply_dtype(mut self, dtype: Option<&str>) -> PyResult<Self> {
        if let Some(dt) = dtype {
            match dt {
                "float64" | "f64" => self.dtype = "float64".to_string(),
                "float32" | "f32" => self.dtype = "float32".to_string(),
                "float16" | "f16" => self.dtype = "float16".to_string(),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unsupported dtype: '{}'",
                        dt
                    )))
                }
            }
        }
        Ok(self)
    }
}

// ── Python API ───────────────────────────────────────────────────────────────

#[pymethods]
impl Array {
    /// Unified Constructor: Detects shape automatically from nested Sequences.
    ///
    /// Examples:
    ///     >>> ra.Array([1, 2, 3])          # 1D array, shape [3]
    ///     >>> ra.Array([[1, 2], [3, 4]])    # 2D array, shape [2, 2]
    ///     >>> ra.Array(rm.Vector([1, 2]))   # From Vector
    #[new]
    #[pyo3(signature = (data, **kwargs))]
    pub fn new(
        data: Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        // Handshake: If passing a Vector, convert instantly
        if let Ok(v) = data.extract::<PyRef<Vector>>() {
            let d = v.with_slice(|s| s.to_vec());
            let mut out = Self::from_flat(d, vec![v.len_internal()]);
            if let Some(kw) = kwargs {
                if let Ok(Some(dt)) = kw.get_item("dtype") {
                    if let Ok(dt_str) = dt.extract::<String>() {
                        out = out.apply_dtype(Some(&dt_str))?;
                    }
                }
            }
            return Ok(out);
        }

        // Handle empty case: distinguish [] from [[]]
        if let Ok(list) = data.downcast::<pyo3::types::PyList>() {
            if list.is_empty() {
                let mut out = Self::from_flat(vec![], vec![0, 0]);
                if let Some(kw) = kwargs {
                    if let Ok(Some(dt)) = kw.get_item("dtype") {
                        if let Ok(dt_str) = dt.extract::<String>() {
                            out = out.apply_dtype(Some(&dt_str))?;
                        }
                    }
                }
                return Ok(out);
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
                                if row.len() != n_cols {
                                    consistent = false;
                                    break;
                                }
                                for j in 0..n_cols {
                                    if let Ok(val_item) = row.get_item(j) {
                                        if let Ok(val) = val_item.extract::<f64>() {
                                            flat.push(val);
                                        } else {
                                            consistent = false;
                                            break;
                                        }
                                    } else {
                                        consistent = false;
                                        break;
                                    }
                                }
                            } else {
                                consistent = false;
                                break;
                            }
                        } else {
                            consistent = false;
                            break;
                        }
                    }

                    if consistent {
                        let mut out = Self::from_flat(flat, vec![n_rows, n_cols]);
                        if let Some(kw) = kwargs {
                            if let Ok(Some(dt)) = kw.get_item("dtype") {
                                out = out.apply_dtype(Some(&dt.extract::<String>()?))?;
                            }
                        }
                        return Ok(out);
                    }
                }
            }
        }

        // Generic Recursive Fallback for N-D
        let mut flat_data = Vec::new();
        let mut shape = Vec::new();

        fn walk(
            obj: &Bound<'_, PyAny>,
            flat: &mut Vec<f64>,
            shape: &mut Vec<usize>,
            depth: usize,
        ) -> PyResult<()> {
            if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
                let len = list.len();
                if shape.len() <= depth {
                    shape.push(len);
                } else if shape[depth] != len {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Inhomogeneous shape in nested list",
                    ));
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
        let mut out = Self::from_flat(flat_data, shape);
        if let Some(kw) = kwargs {
            if let Ok(Some(dt)) = kw.get_item("dtype") {
                out = out.apply_dtype(Some(&dt.extract::<String>()?))?;
            }
        }
        Ok(out)
    }

    // ── Static constructors ───────────────────────────────────────────────

    #[staticmethod]
    #[pyo3(name = "zeros", signature = (*shape, **kwargs))]
    pub fn zeros_py(
        shape: Vec<usize>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let mut arr = Self::zeros_internal(&shape);
        if let Some(kw) = kwargs {
            if let Ok(Some(dt)) = kw.get_item("dtype") {
                arr = arr.apply_dtype(Some(&dt.extract::<String>()?))?;
            }
        }
        Ok(arr)
    }

    /// Create a new N-dimensional Array filled with ones.
    #[staticmethod]
    #[pyo3(name = "ones", signature = (*shape, **kwargs))]
    pub fn ones_py(
        shape: Vec<usize>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let mut arr = Self::ones_internal(&shape);
        if let Some(kw) = kwargs {
            if let Ok(Some(dt)) = kw.get_item("dtype") {
                arr = arr.apply_dtype(Some(&dt.extract::<String>()?))?;
            }
        }
        Ok(arr)
    }

    /// Create a new N-dimensional Array filled with a scalar value.
    ///
    /// Syntax: `Array.full(dim1, dim2, ..., val)`
    #[staticmethod]
    #[pyo3(signature = (*args))]
    pub fn full(args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Self> {
        let n_args = args.len();
        if n_args < 2 {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "full() requires at least 1 dimension and a value",
            ));
        }
        let val = args.get_item(n_args - 1)?.extract::<f64>()?;
        let mut shape = Vec::with_capacity(n_args - 1);
        for i in 0..n_args - 1 {
            shape.push(args.get_item(i)?.extract::<usize>()?);
        }
        Ok(Self::full_internal(&shape, val))
    }

    /// Create a 2D identity matrix of size N x N.
    ///
    /// Examples:
    ///     >>> ra.Array.eye(3)
    ///     Array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    #[staticmethod]
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_flat(data, vec![n, n])
    }

    #[staticmethod]
    #[pyo3(signature = (*shape))]
    pub fn randn(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map_init(thread_rng, |rng, _| rng.sample(StandardNormal))
                .collect()
        } else {
            let mut rng = thread_rng();
            (0..n).map(|_| rng.sample(StandardNormal)).collect()
        };
        Self::from_flat(data, shape)
    }

    #[staticmethod]
    #[pyo3(signature = (*shape))]
    pub fn rand_uniform(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map_init(thread_rng, |rng, _| rng.r#gen::<f64>())
                .collect()
        } else {
            let mut rng = thread_rng();
            (0..n).map(|_| rng.r#gen::<f64>()).collect()
        };
        Self::from_flat(data, shape)
    }

    /// Create a 1D Array with a range of values.
    ///
    /// The values are generated in the interval `[start, stop)` with a step of `step`.
    #[staticmethod]
    pub fn arange(start: f64, stop: f64, step: f64) -> PyResult<Self> {
        if step == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("step=0"));
        }
        let n = ((stop - start) / step).ceil().max(0.0) as usize;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Ok(Self::from_flat(data.clone(), vec![1, data.len()]))
    }

    #[staticmethod]
    #[pyo3(name = "range")]
    pub fn range_py(start: f64, stop: f64, step: f64) -> PyResult<Self> {
        Self::arange(start, stop, step)
    }

    #[staticmethod]
    pub fn linspace(start: f64, stop: f64, n: usize) -> Self {
        if n == 0 {
            return Self::from_flat(vec![], vec![1, 0]);
        }
        if n == 1 {
            return Self::from_flat(vec![start], vec![1, 1]);
        }
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

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    #[pyo3(name = "ndim")]
    pub fn ndim_py(&self) -> usize {
        self.ndim()
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.len()
    }
    pub fn nrows(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape[0]
        }
    }
    pub fn ncols(&self) -> usize {
        if self.shape.len() < 2 {
            1
        } else {
            self.shape[1]
        }
    }

    /// Reshape the array to a new shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let new_n: usize = new_shape.iter().product();
        if new_n != self.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot reshape size {} into {:?}",
                self.len(),
                new_shape
            )));
        }
        if self.is_contiguous() {
            let strides = Self::compute_strides(&new_shape);
            Ok(Array {
                storage: self.storage.clone(),
                shape: new_shape,
                strides,
                offset: self.offset,
                dtype: self.dtype.clone(),
            })
        } else {
            self.to_contiguous().reshape(new_shape)
        }
    }

    /// Return a 1D copy of the array.
    ///
    /// If the array is already contiguous, this is a zero-copy operation.
    pub fn flatten(&self) -> Self {
        let n = self.len();
        if self.is_contiguous() {
            Self {
                storage: self.storage.clone(),
                shape: vec![1, n],
                strides: vec![n as isize, 1],
                offset: self.offset,
                dtype: self.dtype.clone(),
            }
        } else {
            self.to_contiguous().flatten()
        }
    }

    /// Remove single-dimensional entries from the shape of an array.
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().cloned().filter(|&s| s != 1).collect();
        let new_shape = if new_shape.is_empty() {
            vec![1]
        } else {
            new_shape
        };
        let strides = Self::compute_strides(&new_shape);
        Array {
            storage: self.storage.clone(),
            shape: new_shape,
            strides,
            offset: self.offset,
            dtype: self.dtype.clone(),
        }
    }

    /// Insert a new axis at the specified position.
    pub fn expand_dims(&self, axis: usize) -> PyResult<Self> {
        if axis > self.ndim() {
            return Err(pyo3::exceptions::PyValueError::new_err("axis out of range"));
        }
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        let strides = Self::compute_strides(&new_shape);
        Ok(Array {
            storage: self.storage.clone(),
            shape: new_shape,
            strides,
            offset: self.offset,
            dtype: self.dtype.clone(),
        })
    }

    // ── Indexing ──────────────────────────────────────────────────────────

    pub fn __getitem__(&self, py: Python<'_>, key: Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Case 1: Multi-index sequence or Tuple
        let indices: Vec<Bound<'_, PyAny>> = if let Ok(tuple) = key.downcast::<PyTuple>() {
            tuple.iter().collect()
        } else if let Ok(list) = key.downcast::<pyo3::types::PyList>() {
            list.iter().collect()
        } else {
            vec![key.clone()]
        };

        if indices.len() > self.ndim() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Too many indices"));
        }

        let mut is_slice = false;
        let mut slice_ranges = Vec::new();

        for (i, item) in indices.iter().enumerate() {
            let dim_size = self.shape[i];
            if let Ok(slice) = item.downcast::<PySlice>() {
                is_slice = true;
                let indices = slice.indices(dim_size as isize)?;
                let start = indices.start as usize;
                let stop = indices.stop as usize;
                let step = indices.step as usize;
                slice_ranges.push((start, stop, step));
            } else {
                let ty_name = item
                    .get_type()
                    .name()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| "Unknown".to_string());
                let mut idx_val = item.extract::<isize>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "Expected integer or slice, got {}",
                        ty_name
                    ))
                })?;
                if idx_val < 0 {
                    idx_val += dim_size as isize;
                }
                if idx_val < 0 || idx_val >= dim_size as isize {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Index {} out of bounds",
                        idx_val
                    )));
                }
                slice_ranges.push((idx_val as usize, (idx_val + 1) as usize, 1));
            }
        }

        // Fill remaining dimensions with full slices
        for i in indices.len()..self.ndim() {
            slice_ranges.push((0, self.shape[i], 1));
            is_slice = true; // Implicitly slicing if dimensions are omitted?
                             // NumPy behavior: arr[0] on 2D returns a 1D row (a slice).
        }

        if !is_slice && indices.len() == self.ndim() {
            // Scalar return
            let final_indices: Vec<usize> = slice_ranges.iter().map(|r| r.0).collect();
            return Ok(self
                .get(&final_indices)
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }

        // Slice return (Zero-Copy View)
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        let mut new_offset = self.offset;

        for i in 0..self.ndim() {
            let (start, stop, step) = slice_ranges[i];
            let dim_new_size = if stop > start {
                ((stop - start) as f64 / step as f64).ceil() as usize
            } else {
                0
            };

            new_shape.push(dim_new_size);
            new_strides.push(self.strides[i] * step as isize);
            new_offset = (new_offset as isize + start as isize * self.strides[i]) as usize;
        }

        let new_arr = Array {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            dtype: self.dtype.clone(),
        };
        Ok(new_arr.into_pyobject(py)?.into_any().unbind())
    }

    pub fn __setitem__(&mut self, idx: Vec<usize>, val: f64) -> PyResult<()> {
        if idx.len() != self.ndim() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Wrong number of indices",
            ));
        }
        self.make_owned();
        let fi = self.flat_index(&idx);
        match &mut self.storage {
            ArrayStorage::Inline(d, _) => d[fi] = val,
            ArrayStorage::Heap(arc) => {
                std::sync::Arc::get_mut(arc).unwrap()[fi] = val;
            }
        }
        Ok(())
    }

    pub fn get_row(&self, i: usize) -> PyResult<Vec<f64>> {
        if self.ndim() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Not 2-D"));
        }
        if i >= self.nrows() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Row out of bounds"));
        }
        let c = self.ncols();
        Ok((0..c).map(|j| self.get(&[i, j])).collect())
    }

    pub fn get_col(&self, j: usize) -> PyResult<Vec<f64>> {
        if self.ndim() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Not 2-D"));
        }
        if j >= self.ncols() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Col out of bounds"));
        }
        let r = self.nrows();
        Ok((0..r).map(|i| self.get(&[i, j])).collect())
    }

    pub fn slice_rows(&self, start: usize, end: usize) -> PyResult<Self> {
        if self.ndim() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Not 2-D"));
        }
        let end = end.min(self.nrows());
        if start >= end {
            return Err(pyo3::exceptions::PyValueError::new_err("start >= end"));
        }
        let new_offset = self.offset + start * self.strides[0] as usize;
        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;
        Ok(Array {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            dtype: self.dtype.clone(),
        })
    }

    // ── Conversion ────────────────────────────────────────────────────────

    /// Convert the 2D array to a nested Python list (Vec<Vec<f64>>).
    pub fn to_list(&self) -> Vec<Vec<f64>> {
        let (r, c) = (self.nrows(), self.ncols());
        let contig = self.to_contiguous();
        let d = contig.data();
        (0..r)
            .into_par_iter()
            .map(|i| d[i * c..(i + 1) * c].to_vec())
            .collect()
    }

    /// Check if a 2D array is symmetric (A = A^T).
    pub fn is_symmetric(&self) -> bool {
        if self.ndim() != 2 || self.nrows() != self.ncols() {
            return false;
        }
        let n = self.nrows();
        let contig = self.to_contiguous();
        let d = contig.data();
        for i in 0..n {
            for j in 0..i {
                if (d[i * n + j] - d[j * n + i]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Check if a 2D array is positive definite via Cholesky decomposition.
    pub fn is_positive_definite(&self) -> bool {
        self.cholesky_internal().is_ok()
    }

    pub fn normalize(&self, mean: &Bound<'_, PyAny>, std: &Bound<'_, PyAny>) -> PyResult<Self> {
        let centered = self.__sub__(mean)?;
        centered.__truediv__(std)
    }

    pub fn is_prime(&self) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|&x| {
                let n = x.abs().round() as u64;
                if n < 2 {
                    return false;
                }
                if n == 2 {
                    return true;
                }
                if n % 2 == 0 {
                    return false;
                }
                let limit = (n as f64).sqrt() as u64;
                (3..=limit).step_by(2).all(|i| n % i != 0)
            })
            .collect()
    }

    pub fn tolist(&self) -> Vec<Vec<f64>> {
        self.to_list()
    }

    pub fn to_flat_list(&self) -> Vec<f64> {
        self.to_contiguous().data().to_vec()
    }

    /// From flat list + shape
    #[staticmethod]
    pub fn from_list(data: Vec<f64>, shape: Vec<usize>) -> PyResult<Self> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "data len {} != shape product {}",
                data.len(),
                n
            )));
        }
        Ok(Self::from_flat(data, shape))
    }

    /// Create a deep copy of the array.
    pub fn copy(&self) -> Self {
        let data = self.to_contiguous().data().to_vec();
        Self::from_flat(data, self.shape.clone())
    }

    /// Zero-copy (if possible) conversion to Vector
    pub fn into_vector(&self) -> PyResult<Vector> {
        if self.ndim() > 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot convert N-D array to Vector",
            ));
        }
        Ok(Vector::new(self.to_contiguous().data().to_vec()))
    }

    // ── Display ───────────────────────────────────────────────────────────

    pub fn __repr__(&self) -> String {
        let contig = self.to_contiguous();
        let d = contig.data();
        let shape = &self.shape;

        if shape.is_empty() {
            return format!("Array({})", d.first().unwrap_or(&0.0));
        }

        if shape.len() == 1 {
            let n = shape[0];
            if n <= 10 {
                let formatted_vals: Vec<String> = d.iter().map(|v| format!("{v:.4}")).collect();
                return format!("Array([{}])", formatted_vals.join(", "));
            } else {
                return format!(
                    "Array([{}, {}, ..., {}, {}], len={})",
                    d[0],
                    d[1],
                    d[n - 2],
                    d[n - 1],
                    n
                );
            }
        }

        if shape.len() == 2 {
            let rows = shape[0];
            let cols = shape[1];

            let row_limit = if rows > 10 { 3 } else { rows };
            let col_limit = if cols > 10 { 3 } else { cols };

            let mut s = String::from("Array([\n");

            let format_row = |r: usize| -> String {
                let mut row_str = String::from("       [");
                let mut row_vals = Vec::new();
                for c in 0..col_limit {
                    row_vals.push(format!("{:>7.4}", d[r * cols + c]));
                }
                if cols > 10 {
                    row_vals.push("  ...  ".to_string());
                    for c in cols - 3..cols {
                        row_vals.push(format!("{:>7.4}", d[r * cols + c]));
                    }
                }
                row_str.push_str(&row_vals.join(", "));
                row_str.push(']');
                row_str
            };

            for r in 0..row_limit {
                s.push_str(&format_row(r));
                if r < row_limit - 1 || rows > 10 {
                    s.push_str(",\n");
                }
            }

            if rows > 10 {
                s.push_str("       ...,\n");
                for r in rows - 3..rows {
                    s.push_str(&format_row(r));
                    if r < rows - 1 {
                        s.push_str(",\n");
                    }
                }
            }
            s.push_str("])");
            return s;
        }

        // Fallback for >2D or large arrays:
        format!("Array(shape={:?}, type=float64)", shape)
    }

    pub fn __len__(&self) -> usize {
        self.nrows()
    }
}

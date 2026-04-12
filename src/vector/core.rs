use pyo3::prelude::*;
use pyo3::types::PySlice;
use rayon::prelude::*;
use std::sync::Arc;
use crate::array::Array;

// ---------------------------------------------------------------------------
// Parallelism thresholds (empirically calibrated, tuneable)
//
//   INLINE_MAX  : vectors at or below this count use stack storage and
//                 purely serial scalar loops.  32 f64 = 256 bytes, fits in
//                 L1 cache.  Never touch Rayon.
//
//   PAR_THRESHOLD : vectors at or above this count use Rayon par_iter.
//                 Below this, serial iteration beats Rayon even for
//                 transcendental ops (sin, exp) due to thread-spawn overhead.
//                 Rayon's own docs cite this as ~10_000 for trivial work;
//                 we use 8_192 as a conservative cross-over.
// ---------------------------------------------------------------------------
const INLINE_MAX: usize = 32;
const PAR_THRESHOLD: usize = 8_192;

// ---------------------------------------------------------------------------
// VectorStorage
//
// Three-tier:
//   Inline  — stack array, zero heap allocation, zero GC pressure
//   Heap    — Arc<Vec<f64>>, shared-ownership, clone-on-write for mutation
//
// The Inline tier is maintained across operations: map_inline keeps small
// results small.  Only when a result genuinely exceeds INLINE_MAX does it
// graduate to Heap.
// ---------------------------------------------------------------------------
/// Internal storage representation for the Vector engine.
///
/// Implements a three-tier memory model:
/// - `Inline`: Stack-allocated for zero-overhead math on small vectors (<32 elements).
/// - `Heap`: Shared memory via Arc<Vec<f64>> for large datasets.
#[derive(Clone)]
pub enum VectorStorage {
    Inline([f64; INLINE_MAX], usize),
    Heap(Arc<Vec<f64>>),
}

/// High-performance 1D vectorized container.
///
/// `Vector` is optimized for linear scans and high-speed element-wise math.
/// It uses a tiered storage engine to bypass heap allocations for small
/// data vectors, ensuring L1-cache speed for hot-path calculations.
///
/// Use `Vector` when dealing specifically with 1D signals or column data
/// where the overhead of N-D coordinate mapping is unwanted.
#[pyclass]
#[derive(Clone)]
pub struct Vector {
    pub storage: VectorStorage,
}

// ---------------------------------------------------------------------------
// Internal Rust API — no PyO3 annotations, used by ops.rs and pipeline bridge
// ---------------------------------------------------------------------------
impl Vector {
    // --- Construction helpers ---

    pub fn new(data: Vec<f64>) -> Self {
        let n = data.len();
        if n <= INLINE_MAX {
            let mut arr = [0.0_f64; INLINE_MAX];
            arr[..n].copy_from_slice(&data);
            Vector { storage: VectorStorage::Inline(arr, n) }
        } else {
            Vector { storage: VectorStorage::Heap(Arc::new(data)) }
        }
    }

    pub fn from_arc(arc: Arc<Vec<f64>>) -> Self {
        let n = arc.len();
        if n <= INLINE_MAX {
            let mut arr = [0.0_f64; INLINE_MAX];
            arr[..n].copy_from_slice(&arc);
            Vector { storage: VectorStorage::Inline(arr, n) }
        } else {
            Vector { storage: VectorStorage::Heap(arc) }
        }
    }

    pub fn zeros_internal(n: usize) -> Self {
        if n <= INLINE_MAX {
            Vector { storage: VectorStorage::Inline([0.0; INLINE_MAX], n) }
        } else {
            Vector::new(vec![0.0; n])
        }
    }

    pub fn ones_internal(n: usize) -> Self {
        if n <= INLINE_MAX {
            let mut arr = [0.0_f64; INLINE_MAX];
            arr[..n].fill(1.0);
            Vector { storage: VectorStorage::Inline(arr, n) }
        } else {
            Vector::new(vec![1.0; n])
        }
    }

    pub fn full_internal(n: usize, val: f64) -> Self {
        if n <= INLINE_MAX {
            let mut arr = [0.0_f64; INLINE_MAX];
            arr[..n].fill(val);
            Vector { storage: VectorStorage::Inline(arr, n) }
        } else {
            Vector::new(vec![val; n])
        }
    }

    // --- Slice access ---

    #[inline(always)]
    pub fn len_internal(&self) -> usize {
        match &self.storage {
            VectorStorage::Inline(_, n) => *n,
            VectorStorage::Heap(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len_internal() == 0
    }

    /// Borrow the underlying slice, run f, return result.
    #[inline(always)]
    pub fn with_slice<R, F: FnOnce(&[f64]) -> R>(&self, f: F) -> R {
        match &self.storage {
            VectorStorage::Inline(arr, n) => f(&arr[..*n]),
            VectorStorage::Heap(v) => f(v.as_slice()),
        }
    }

    // --- Elementwise map — respects tiers ---
    //
    // Inline: pure scalar loop, stays Inline if result fits.
    // Heap small (< PAR_THRESHOLD): serial iterator.
    // Heap large (>= PAR_THRESHOLD): Rayon par_iter.
    pub fn map_internal<F>(&self, f: F) -> Vector
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        match &self.storage {
            VectorStorage::Inline(arr, n) => {
                let mut out = [0.0_f64; INLINE_MAX];
                for i in 0..*n {
                    out[i] = f(arr[i]);
                }
                Vector { storage: VectorStorage::Inline(out, *n) }
            }
            VectorStorage::Heap(v) => {
                let n = v.len();
                let data: Vec<f64> = if n >= PAR_THRESHOLD {
                    v.par_iter().map(|&x| f(x)).collect()
                } else {
                    v.iter().map(|&x| f(x)).collect()
                };
                Vector::from_arc(Arc::new(data))
            }
        }
    }

    /// Elementwise binary operation on two vectors of equal length.
    pub fn zip_map_internal<F>(&self, other: &Vector, f: F) -> PyResult<Vector>
    where
        F: Fn(f64, f64) -> f64 + Sync + Send,
    {
        let n = self.len_internal();
        if n != other.len_internal() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "length mismatch: {} vs {}",
                n, other.len_internal()
            )));
        }
        let data = self.with_slice(|s1| {
            other.with_slice(|s2| {
                if n >= PAR_THRESHOLD {
                    s1.par_iter().zip(s2.par_iter()).map(|(&a, &b)| f(a, b)).collect::<Vec<f64>>()
                } else {
                    s1.iter().zip(s2.iter()).map(|(&a, &b)| f(a, b)).collect::<Vec<f64>>()
                }
            })
        });
        Ok(Vector::new(data))
    }

    // --- Reduction helpers (serial for small, parallel for large) ---

    pub fn reduce_serial<T, F, G>(&self, init: T, fold: F, _combine: G) -> T
    where
        T: Copy,
        F: Fn(T, f64) -> T,
        G: Fn(T, T) -> T,
    {
        self.with_slice(|s| s.iter().fold(init, |acc, &x| fold(acc, x)))
    }

    /// Kahan compensated sum for numerical accuracy on large vectors.
    /// Error is O(ε) rather than O(N·ε) for naive summation.
    pub fn kahan_sum(&self) -> f64 {
        self.with_slice(|s| {
            let n = s.len();
            if n >= PAR_THRESHOLD {
                // Parallel: chunk-sum then Kahan-combine
                // (full parallel Kahan is complex; we do parallel chunk sums
                //  then a serial Kahan pass over chunk totals — good enough)
                let chunk = (n + rayon::current_num_threads() - 1) / rayon::current_num_threads();
                let chunk_sums: Vec<f64> = s
                    .par_chunks(chunk.max(1))
                    .map(|c| {
                        let mut sum = 0.0_f64;
                        let mut comp = 0.0_f64;
                        for &x in c {
                            let y = x - comp;
                            let t = sum + y;
                            comp = (t - sum) - y;
                            sum = t;
                        }
                        sum
                    })
                    .collect();
                // Serial Kahan over chunk sums
                let mut sum = 0.0_f64;
                let mut comp = 0.0_f64;
                for x in chunk_sums {
                    let y = x - comp;
                    let t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }
                sum
            } else {
                let mut sum = 0.0_f64;
                let mut comp = 0.0_f64;
                for &x in s {
                    let y = x - comp;
                    let t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }
                sum
            }
        })
    }

    /// Welford single-pass mean and M2 (for variance).
    /// Returns (mean, M2, count).
    pub fn welford(&self) -> (f64, f64, usize) {
        self.with_slice(|s| {
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            let mut count = 0_usize;
            for &x in s {
                count += 1;
                let delta = x - mean;
                mean += delta / count as f64;
                m2 += delta * (x - mean);
            }
            (mean, m2, count)
        })
    }
    pub fn moments(&self) -> (f64, f64, usize) {
        let (m, m2, n) = self.welford();
        if n < 2 { return (m, 0.0, n); }
        (m, (m2 / (n as f64 - 1.0)).sqrt(), n)
    }
}

// ---------------------------------------------------------------------------
// Python-exposed methods
// ---------------------------------------------------------------------------
#[pymethods]
impl Vector {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new Vector from a sequence of floats or an Array.
    ///
    /// Accepts any Python sequence (list, tuple) or an RMath Array
    /// (if the array is 1-D or 2-D with one dimension of size 1).
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector([1.0, 2.0, 3.0])
    ///     Vector([1.0, 2.0, 3.0])
    #[new]
    pub fn py_new(data: Bound<'_, PyAny>) -> PyResult<Self> {
        // Handshake: If passing an Array, convert if 1-D or 2-D (1xN/Nx1)
        if let Ok(a) = data.extract::<PyRef<Array>>() {
            if a.ndim() > 2 {
                return Err(pyo3::exceptions::PyValueError::new_err("Cannot create Vector from N-D array (ndim > 2)"));
            }
            return Ok(Vector::new(a.data().to_vec()));
        }

        // Standard list/sequence path
        let d: Vec<f64> = data.extract()?;
        Ok(Vector::new(d))
    }

    /// Create a Vector filled with zeros.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector.zeros(3)
    ///     Vector([0.0, 0.0, 0.0])
    #[staticmethod]
    pub fn zeros(n: usize) -> Self { Vector::zeros_internal(n) }

    /// Create a Vector filled with ones.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector.ones(3)
    ///     Vector([1.0, 1.0, 1.0])
    #[staticmethod]
    pub fn ones(n: usize) -> Self { Vector::ones_internal(n) }

    /// Create a Vector filled with a specific value.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector.full(3, 42.0)
    ///     Vector([42.0, 42.0, 42.0])
    #[staticmethod]
    pub fn full(n: usize, val: f64) -> Self { Vector::full_internal(n, val) }

    #[staticmethod]
    pub fn random(n: usize) -> Self {
        super::ops::vector_rand(n)
    }

    #[staticmethod]
    pub fn randn(n: usize) -> Self {
        super::ops::vector_randn(n)
    }

    #[staticmethod]
    pub fn rand_seeded(n: usize, seed: u64) -> Self {
        super::ops::vector_rand_seeded(n, seed)
    }

    /// Create a Vector with a range of values.
    ///
    /// Usage: `Vector.arange(stop)` or `Vector.arange(start, stop, step)`.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector.arange(5)
    ///     Vector([0.0, 1.0, 2.0, 3.0, 4.0])
    ///     >>> Vector.arange(1, 4, 0.5)
    ///     Vector([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    #[staticmethod]
    #[pyo3(signature = (start, stop = None, step = 1.0))]
    pub fn arange(start: f64, stop: Option<f64>, step: f64) -> Self {
        let (s, e) = match stop { Some(v) => (start, v), None => (0.0, start) };
        if step == 0.0 { return Vector::new(vec![]); }
        let n_f = ((e - s) / step).ceil();
        if n_f <= 0.0 { return Vector::new(vec![]); }
        let n = n_f as usize;
        let data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter().map(|i| s + i as f64 * step).collect()
        } else {
            (0..n).map(|i| s + i as f64 * step).collect()
        };
        Vector::new(data)
    }

    #[staticmethod]
    #[pyo3(name = "range", signature = (start, stop = None, step = 1.0))]
    pub fn range_py(start: f64, stop: Option<f64>, step: f64) -> Self {
        Self::arange(start, stop, step)
    }

    /// Create a Vector with `num` evenly spaced values from `start` to `stop`.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector.linspace(0, 1, 5)
    ///     Vector([0.0, 0.25, 0.5, 0.75, 1.0])
    #[staticmethod]
    pub fn linspace(start: f64, stop: f64, num: usize) -> Self {
        if num == 0 { return Vector::new(vec![]); }
        if num == 1 { return Vector::new(vec![start]); }
        let step = (stop - start) / (num - 1) as f64;
        let data: Vec<f64> = if num >= PAR_THRESHOLD {
            (0..num).into_par_iter().map(|i| start + i as f64 * step).collect()
        } else {
            (0..num).map(|i| start + i as f64 * step).collect()
        };
        Vector::new(data)
    }

    /// Closed-form arithmetic sum of a range.
    ///
    /// Calculates the sum of `arange(start, stop, step)` without allocation.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector.sum_range(100)
    ///     4950.0
    #[staticmethod]
    #[pyo3(signature = (start_or_stop, stop = None, step = 1.0))]
    pub fn sum_range(start_or_stop: f64, stop: Option<f64>, step: f64) -> f64 {
        let (start, stop) = match stop { Some(s) => (start_or_stop, s), None => (0.0, start_or_stop) };
        if step == 0.0 { return 0.0; }
        let n = ((stop - start) / step).ceil().max(0.0);
        if n <= 0.0 { return 0.0; }
        let last = start + (n - 1.0) * step;
        (n / 2.0) * (start + last)
    }

    // -----------------------------------------------------------------------
    // Python sequence protocol
    // -----------------------------------------------------------------------

    pub fn __len__(&self) -> usize { self.len_internal() }

    pub fn __iter__(&self) -> VectorIter {
        match &self.storage {
            // Heap: O(1) Arc reference-count bump — no memcopy regardless of N.
            VectorStorage::Heap(arc) => VectorIter {
                source: IterSource::Heap(Arc::clone(arc)),
                pos: 0,
            },
            // Inline: at most 32 × f64 = 256 bytes — copy is negligible.
            VectorStorage::Inline(arr, n) => {
                let mut buf = [0.0_f64; INLINE_MAX];
                buf[..*n].copy_from_slice(&arr[..*n]);
                VectorIter {
                    source: IterSource::Inline(buf, *n),
                    pos: 0,
                }
            }
        }
    }

    pub fn __contains__(&self, val: f64) -> bool {
        self.with_slice(|s| s.iter().any(|&x| x == val))
    }

    pub fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Integer indexing
            if let Ok(i) = index.extract::<isize>() {
                let len = self.len_internal() as isize;
                let idx = if i < 0 { i + len } else { i };
                if idx < 0 || idx >= len {
                    return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
                }
                let val = self.with_slice(|s| s[idx as usize]);
                return Ok(crate::scalar::Scalar(val).into_pyobject(py)?.into_any().unbind());
            }
            // Slice indexing
            if let Ok(sl) = index.downcast::<PySlice>() {
                let len = self.len_internal();
                let indices = sl.indices(len as isize)?;
                let start = indices.start as usize;
                let stop  = indices.stop as usize;
                let step  = indices.step as usize;
                let data: Vec<f64> = self.with_slice(|s| {
                    (start..stop).step_by(step.max(1)).map(|i| s[i]).collect()
                });
                return Ok(Vector::new(data).into_pyobject(py)?.into_any().unbind());
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "__getitem__ requires int or slice",
            ))
        })
    }

    pub fn __setitem__(&mut self, index: isize, value: f64) -> PyResult<()> {
        let len = self.len_internal() as isize;
        let idx = if index < 0 { index + len } else { index };
        if idx < 0 || idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err("index out of range"));
        }
        match &mut self.storage {
            VectorStorage::Inline(arr, _) => { arr[idx as usize] = value; }
            VectorStorage::Heap(arc) => { Arc::make_mut(arc)[idx as usize] = value; }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Representation
    // -----------------------------------------------------------------------

    /// Standard string representation.
    pub fn __repr__(&self) -> String {
        self.with_slice(|s| {
            if s.len() <= 6 {
                format!("Vector({:?})", s)
            } else {
                format!("Vector([{}, {}, ..., {}], len={})", s[0], s[1], s[s.len()-1], s.len())
            }
        })
    }

    /// User-friendly string representation.
    pub fn __str__(&self) -> String { self.__repr__() }

    // -----------------------------------------------------------------------
    // Conversion
    // -----------------------------------------------------------------------

    /// Convert Vector elements to a Python list.
    pub fn to_list(&self) -> Vec<f64> { self.with_slice(|s| s.to_vec()) }
    /// Alias for `to_list()`.
    pub fn tolist(&self) -> Vec<f64> { self.to_list() }

    /// Create a zero-copy LazyPipeline from this Vector.
    ///
    /// Useful for chaining multiple operations without intermediate allocations.
    pub fn lazy(&self) -> crate::scalar::LazyPipeline {
        match &self.storage {
            VectorStorage::Heap(arc) => crate::scalar::from_shared_buffer(Arc::clone(arc)),
            VectorStorage::Inline(arr, n) => {
                crate::scalar::from_shared_buffer(Arc::new(arr[..*n].to_vec()))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Equality
    // -----------------------------------------------------------------------

    /// Elementwise equality → Python list of bool.
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
        if let Ok(s) = other.extract::<f64>() {
            return Ok(self.with_slice(|sl| sl.iter().map(|&x| x == s).collect()));
        }
        if let Ok(v) = other.extract::<PyRef<Vector>>() {
            let n = self.len_internal();
            if n != v.len_internal() {
                return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
            }
            return Ok(self.with_slice(|s1| v.with_slice(|s2| {
                s1.iter().zip(s2.iter()).map(|(a, b)| a == b).collect()
            })));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    // -----------------------------------------------------------------------
    // Arithmetic operators
    // -----------------------------------------------------------------------

    pub fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_internal(|x| x + s)); }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() { return self.zip_map_internal(&v, |a,b| a+b); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        // addition is commutative
        self.__add__(lhs)
    }

    pub fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_internal(|x| x - s)); }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() { return self.zip_map_internal(&v, |a,b| a-b); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        // lhs - self
        if let Ok(s) = lhs.extract::<f64>() { return Ok(self.map_internal(|x| s - x)); }
        if let Ok(v) = lhs.extract::<PyRef<Vector>>() { return v.zip_map_internal(self, |a,b| a-b); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_internal(|x| x * s)); }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() { return self.zip_map_internal(&v, |a,b| a*b); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        self.__mul__(lhs)
    }

    pub fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero")); }
            return Ok(self.map_internal(|x| x / s));
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            // elementwise: 0-divisors produce INFINITY (documented IEEE behaviour)
            return self.zip_map_internal(&v, |a,b| a/b);
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        // lhs / self
        if let Ok(s) = lhs.extract::<f64>() { return Ok(self.map_internal(|x| s / x)); }
        if let Ok(v) = lhs.extract::<PyRef<Vector>>() { return v.zip_map_internal(self, |a,b| a/b); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __floordiv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero")); }
            return Ok(self.map_internal(|x| (x/s).floor()));
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() { return self.zip_map_internal(&v, |a,b| (a/b).floor()); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __mod__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Vector> {
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("modulo by zero")); }
            return Ok(self.map_internal(|x| x % s));
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() { return self.zip_map_internal(&v, |a,b| a%b); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn __pow__(&self, exp: &Bound<'_, PyAny>, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Vector> {
        if let Ok(s) = exp.extract::<f64>() { return Ok(self.map_internal(|x| x.powf(s))); }
        if let Ok(v) = exp.extract::<PyRef<Vector>>() { return self.zip_map_internal(&v, |a,b| a.powf(b)); }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn pow(&self, exp: &Bound<'_, PyAny>) -> PyResult<Vector> {
        self.__pow__(exp, None)
    }

    pub fn __neg__(&self) -> Vector { self.map_internal(|x| -x) }
    pub fn __pos__(&self) -> Vector { self.clone() }
    pub fn __abs__(&self) -> Vector { self.map_internal(|x| x.abs()) }

    /// Zero-copy conversion to Array
    pub fn into_array(&self) -> Array {
        Array::from_flat(self.to_list(), vec![self.len_internal()])
    }

    pub fn add(&self, other: &Bound<'_, PyAny>) -> PyResult<Vector> { self.__add__(other) }
    pub fn sub(&self, other: &Bound<'_, PyAny>) -> PyResult<Vector> { self.__sub__(other) }
    pub fn mul(&self, other: &Bound<'_, PyAny>) -> PyResult<Vector> { self.__mul__(other) }
    pub fn div(&self, other: &Bound<'_, PyAny>) -> PyResult<Vector> { self.__truediv__(other) }

    pub fn add_vec(&self, other: &Vector) -> PyResult<Vector> { self.zip_map_internal(other, |a, b| a + b) }
    pub fn sub_vec(&self, other: &Vector) -> PyResult<Vector> { self.zip_map_internal(other, |a, b| a - b) }
    pub fn mul_vec(&self, other: &Vector) -> PyResult<Vector> { self.zip_map_internal(other, |a, b| a * b) }
    pub fn div_vec(&self, other: &Vector) -> PyResult<Vector> { self.zip_map_internal(other, |a, b| a / b) }

    pub fn add_scalar(&self, s: f64) -> Vector { self.map_internal(|x| x + s) }
    pub fn sub_scalar(&self, s: f64) -> Vector { self.map_internal(|x| x - s) }
    pub fn mul_scalar(&self, s: f64) -> Vector { self.map_internal(|x| x * s) }
    pub fn div_scalar(&self, s: f64) -> PyResult<Vector> {
        if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("division by zero")); }
        Ok(self.map_internal(|x| x / s))
    }

    // -----------------------------------------------------------------------
    // Reductions — all return f64 (NaN for empty), never Option
    // -----------------------------------------------------------------------

    /// Sum all elements using Kahan compensated summation.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector([1, 2, 3]).sum()
    ///     6.0
    pub fn sum(&self) -> f64 {
        if self.is_empty() { return 0.0; }
        self.kahan_sum()
    }

    pub fn prod(&self) -> f64 {
        if self.is_empty() { return 1.0; }
        self.with_slice(|s| {
            if s.len() >= PAR_THRESHOLD {
                s.par_iter().cloned().reduce(|| 1.0, |a, b| a * b)
            } else {
                s.iter().cloned().product()
            }
        })
    }

    /// Arithmetic mean. Returns `NaN` for an empty vector.
    pub fn mean(&self) -> f64 {
        let n = self.len_internal();
        if n == 0 { return f64::NAN; }
        self.sum() / n as f64
    }

    /// Minimum value.  Returns NaN for empty vector.
    pub fn min(&self) -> f64 {
        if self.is_empty() { return f64::NAN; }
        self.with_slice(|s| {
            if s.len() >= PAR_THRESHOLD {
                s.par_iter().cloned().reduce(|| f64::INFINITY, f64::min)
            } else {
                s.iter().cloned().fold(f64::INFINITY, f64::min)
            }
        })
    }

    /// Maximum value.  Returns NaN for empty vector.
    pub fn max(&self) -> f64 {
        if self.is_empty() { return f64::NAN; }
        self.with_slice(|s| {
            if s.len() >= PAR_THRESHOLD {
                s.par_iter().cloned().reduce(|| f64::NEG_INFINITY, f64::max)
            } else {
                s.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
        })
    }

    /// Index of the minimum element.  Returns -1 for empty.
    pub fn argmin(&self) -> isize {
        if self.is_empty() { return -1; }
        self.with_slice(|s| {
            s.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as isize)
                .unwrap_or(-1)
        })
    }

    /// Index of the maximum element.  Returns -1 for empty.
    pub fn argmax(&self) -> isize {
        if self.is_empty() { return -1; }
        self.with_slice(|s| {
            s.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as isize)
                .unwrap_or(-1)
        })
    }

    /// Sample variance (Bessel correction, n-1).  NaN for n < 2.
    pub fn standardize(&self) -> Self {
        let (m, s, _) = self.moments();
        if s == 0.0 {
            self.clone()
        } else {
            self.sub_scalar(m).div_scalar(s).unwrap()
        }
    }

    pub fn softmax(&self) -> Self {
        self.with_slice(|s| {
            if s.is_empty() { return Self::zeros(0); }
            let max_val = s.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f64> = s.par_iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exps: f64 = exps.par_iter().sum();
            if sum_exps == 0.0 {
                Self::new(vec![1.0 / s.len() as f64; s.len()])
            } else {
                Self::new(exps).div_scalar(sum_exps).unwrap()
            }
        })
    }
    /// Sample variance (Bessel's correction n-1). Returns `NaN` for n < 2.
    ///
    /// Examples:
    ///     >>> from rmath import Vector
    ///     >>> Vector([1, 2, 3]).variance()
    ///     1.0
    pub fn variance(&self) -> f64 {
        let (_, m2, n) = self.welford();
        if n < 2 { return f64::NAN; }
        m2 / (n - 1) as f64
    }

    /// Population variance (divided by n). Returns `NaN` for an empty vector.
    pub fn pop_variance(&self) -> f64 {
        let (_, m2, n) = self.welford();
        if n == 0 { return f64::NAN; }
        m2 / n as f64
    }

    /// Sample standard deviation.
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }

    /// Population standard deviation.
    pub fn pop_std_dev(&self) -> f64 { self.pop_variance().sqrt() }

    /// Median.  Returns NaN for empty.  Does not require sorted input.
    pub fn median(&self) -> f64 {
        let n = self.len_internal();
        if n == 0 { return f64::NAN; }
        let mut s = self.with_slice(|sl| sl.to_vec());
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        }
    }

    /// q-th percentile (0–100).  Linear interpolation.
    pub fn percentile(&self, q: f64) -> PyResult<f64> {
        if q < 0.0 || q > 100.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must be in [0, 100]"));
        }
        let n = self.len_internal();
        if n == 0 { return Ok(f64::NAN); }
        let mut s = self.with_slice(|sl| sl.to_vec());
        s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pos = (q / 100.0) * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        Ok(s[lo] + frac * (s[hi] - s[lo]))
    }

    /// L2 (Euclidean) norm: sqrt(sum(x²)).
    pub fn norm(&self) -> f64 {
        self.with_slice(|s| {
            let sq: f64 = if s.len() >= PAR_THRESHOLD {
                s.par_iter().map(|x| x * x).sum()
            } else {
                s.iter().map(|x| x * x).sum()
            };
            sq.sqrt()
        })
    }

    /// L1 (Manhattan) norm: sum(|x|).
    pub fn norm_l1(&self) -> f64 {
        self.with_slice(|s| {
            if s.len() >= PAR_THRESHOLD {
                s.par_iter().map(|x| x.abs()).sum()
            } else {
                s.iter().map(|x| x.abs()).sum()
            }
        })
    }

    /// L-infinity (Chebyshev) norm: max(|x|).
    pub fn norm_inf(&self) -> f64 {
        self.with_slice(|s| {
            s.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
        })
    }

    /// General Lp norm: (sum(|x|^p))^(1/p).
    pub fn norm_lp(&self, p: f64) -> PyResult<f64> {
        if p <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("p must be > 0"));
        }
        Ok(self.with_slice(|s| {
            let sum: f64 = s.iter().map(|x| x.abs().powf(p)).sum();
            sum.powf(1.0 / p)
        }))
    }

    /// Dot product.
    pub fn dot(&self, other: &Vector) -> PyResult<f64> {
        let n = self.len_internal();
        if n != other.len_internal() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dot: length mismatch {} vs {}", n, other.len_internal()
            )));
        }
        Ok(self.with_slice(|s1| other.with_slice(|s2| {
            if n >= PAR_THRESHOLD {
                s1.par_iter().zip(s2.par_iter()).map(|(a,b)| a*b).sum()
            } else {
                s1.iter().zip(s2.iter()).map(|(a,b)| a*b).sum()
            }
        })))
    }

    // -----------------------------------------------------------------------
    // Predicates — return Vec<bool>
    // -----------------------------------------------------------------------

    /// Returns `True` if any element is non-zero.
    pub fn any(&self) -> bool { self.with_slice(|s| s.iter().any(|x| *x != 0.0)) }
    /// Returns `True` if all elements are non-zero.
    pub fn all(&self) -> bool { self.with_slice(|s| s.iter().all(|x| *x != 0.0)) }
    /// Returns a boolean mask: `True` if element is NaN.
    pub fn isnan(&self)    -> Vec<bool> { self.with_slice(|s| s.iter().map(|x| x.is_nan()).collect()) }
    /// Returns a boolean mask: `True` if element is finite.
    pub fn isfinite(&self) -> Vec<bool> { self.with_slice(|s| s.iter().map(|x| x.is_finite()).collect()) }
    /// Returns a boolean mask: `True` if element is infinite.
    pub fn isinf(&self)    -> Vec<bool> { self.with_slice(|s| s.iter().map(|x| x.is_infinite()).collect()) }
    /// Returns a boolean mask: `True` if element is an integer.
    pub fn is_integer(&self) -> Vec<bool> { self.with_slice(|s| s.iter().map(|x| x.fract() == 0.0).collect()) }
    /// Returns a boolean mask: `True` if element is a prime number.
    ///
    /// Rounds each element to the nearest integer for the test.
    pub fn is_prime(&self) -> Vec<bool> {
        self.with_slice(|s| s.iter().map(|&x| {
            let n = x.abs().round() as u64;
            if n < 2 { return false; }
            if n == 2 { return true; }
            if n % 2 == 0 { return false; }
            let limit = (n as f64).sqrt() as u64;
            (3..=limit).step_by(2).all(|i| n % i != 0)
        }).collect())
    }

    // -----------------------------------------------------------------------
    // Filtering and selection
    // -----------------------------------------------------------------------

    /// Return elements where mask[i] is True.
    ///
    /// The mask can be a `Vector` (where non-zero means True) or a Python list of bools.
    ///
    /// Examples:
    ///     >>> v = Vector([1, 2, 3, 4])
    ///     >>> v.filter_by_mask([True, False, True, False])
    ///     Vector([1.0, 3.0])
    pub fn filter_by_mask(&self, mask: &Bound<'_, PyAny>) -> PyResult<Vector> {
        let n = self.len_internal();
        // Fast path: Vector mask (stays in Rust, no Python bool extraction)
        if let Ok(vmask) = mask.extract::<PyRef<Vector>>() {
            if vmask.len_internal() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "mask length must match vector length"
                ));
            }
            let data = self.with_slice(|s| {
                vmask.with_slice(|m| {
                    s.iter().zip(m.iter())
                        .filter(|&(_, &mv)| mv != 0.0)
                        .map(|(&v, _)| v)
                        .collect()
                })
            });
            return Ok(Vector::new(data));
        }
        // Fallback: Python list of bools
        let bool_mask: Vec<bool> = mask.extract()?;
        if bool_mask.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mask length must match vector length"
            ));
        }
        let data = self.with_slice(|s| {
            s.iter().zip(bool_mask.iter()).filter(|&(_, &m)| m).map(|(&v, _)| v).collect()
        });
        Ok(Vector::new(data))
    }

    /// Generate a mask Vector: 1.0 where self[i] > threshold, 0.0 otherwise.
    /// Stays in Rust — use with filter_by_mask for zero-extraction filtering.
    pub fn gt_mask(&self, threshold: f64) -> Vector {
        self.map_internal(|x| if x > threshold { 1.0 } else { 0.0 })
    }

    /// Generate a mask Vector: 1.0 where self[i] < threshold, 0.0 otherwise.
    pub fn lt_mask(&self, threshold: f64) -> Vector {
        self.map_internal(|x| if x < threshold { 1.0 } else { 0.0 })
    }

    /// Generate a mask Vector: 1.0 where self[i] == value, 0.0 otherwise.
    pub fn eq_mask(&self, value: f64) -> Vector {
        self.map_internal(|x| if x == value { 1.0 } else { 0.0 })
    }

    /// Return elements greater than threshold.
    pub fn filter_gt(&self, threshold: f64) -> Vector {
        Vector::new(self.with_slice(|s| s.iter().filter(|&&x| x > threshold).cloned().collect()))
    }

    /// Return elements less than threshold.
    pub fn filter_lt(&self, threshold: f64) -> Vector {
        Vector::new(self.with_slice(|s| s.iter().filter(|&&x| x < threshold).cloned().collect()))
    }

    /// Elementwise selection: where mask[i] → self[i], else other[i].
    pub fn where_(&self, mask: Vec<bool>, other: &Vector) -> PyResult<Vector> {
        let n = self.len_internal();
        if mask.len() != n || other.len_internal() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mask and other must have the same length as self"
            ));
        }
        let data = self.with_slice(|s1| other.with_slice(|s2| {
            s1.iter().zip(s2.iter()).zip(mask.iter())
                .map(|((&a, &b), &m)| if m { a } else { b })
                .collect()
        }));
        Ok(Vector::new(data))
    }

    // -----------------------------------------------------------------------
    // Sorting and reordering
    // -----------------------------------------------------------------------

    /// Return a sorted copy of the vector (ascending).
    pub fn sort(&self) -> Vector {
        let mut v = self.with_slice(|s| s.to_vec());
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Vector::new(v)
    }

    /// Return a sorted copy of the vector (descending).
    pub fn sort_desc(&self) -> Vector {
        let mut v = self.with_slice(|s| s.to_vec());
        v.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Vector::new(v)
    }

    /// Return argsort indices (ascending order).
    pub fn argsort(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.len_internal()).collect();
        self.with_slice(|s| {
            idx.sort_unstable_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap_or(std::cmp::Ordering::Equal));
        });
        idx
    }

    /// Reverse order.
    pub fn reverse(&self) -> Vector {
        let mut v = self.with_slice(|s| s.to_vec());
        v.reverse();
        Vector::new(v)
    }

    /// Remove duplicate values (order-preserving).
    pub fn unique(&self) -> Vector {
        let mut seen = std::collections::HashSet::new();
        let data = self.with_slice(|s| {
            s.iter().filter(|&&x| {
                let bits = x.to_bits();
                seen.insert(bits)
            }).cloned().collect()
        });
        Vector::new(data)
    }

    // -----------------------------------------------------------------------
    // Cumulative operations
    // -----------------------------------------------------------------------

    /// Prefix sum (cumulative sum).
    ///
    /// Examples:
    ///     >>> Vector([1, 2, 3]).cumsum()
    ///     Vector([1.0, 3.0, 6.0])
    pub fn cumsum(&self) -> Vector {
        let data = self.with_slice(|s| {
            let mut acc = 0.0_f64;
            s.iter().map(|&x| { acc += x; acc }).collect()
        });
        Vector::new(data)
    }

    /// Prefix product (cumulative product).
    pub fn cumprod(&self) -> Vector {
        let data = self.with_slice(|s| {
            let mut acc = 1.0_f64;
            s.iter().map(|&x| { acc *= x; acc }).collect()
        });
        Vector::new(data)
    }

    /// Finite differences (first-order): diff[i] = v[i+1] - v[i].
    /// Result has length n-1.
    pub fn diff(&self) -> Vector {
        let data = self.with_slice(|s| {
            if s.len() < 2 { return vec![]; }
            s.windows(2).map(|w| w[1] - w[0]).collect()
        });
        Vector::new(data)
    }

    // -----------------------------------------------------------------------
    // Shaping / slicing
    // -----------------------------------------------------------------------

    /// First n elements.
    pub fn head(&self, n: usize) -> Vector {
        let n = n.min(self.len_internal());
        Vector::new(self.with_slice(|s| s[..n].to_vec()))
    }

    /// Last n elements.
    pub fn tail(&self, n: usize) -> Vector {
        let len = self.len_internal();
        let n = n.min(len);
        Vector::new(self.with_slice(|s| s[len - n..].to_vec()))
    }

    /// Split into chunks of size `chunk_size`.  Last chunk may be smaller.
    pub fn chunks(&self, chunk_size: usize) -> PyResult<Vec<Vector>> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("chunk_size must be > 0"));
        }
        Ok(self.with_slice(|s| {
            s.chunks(chunk_size).map(|c| Vector::new(c.to_vec())).collect()
        }))
    }

    /// Append a single value, returning a new Vector.
    pub fn append(&self, val: f64) -> Vector {
        let mut v = self.with_slice(|s| s.to_vec());
        v.push(val);
        Vector::new(v)
    }

    /// Concatenate another Vector, returning a new Vector.
    pub fn extend(&self, other: &Vector) -> Vector {
        let mut v = self.with_slice(|s| s.to_vec());
        other.with_slice(|s| v.extend_from_slice(s));
        Vector::new(v)
    }
}

#[pymethods]
impl Vector {
    // -----------------------------------------------------------------------
    // Elementwise math (Python-exposed)
    // -----------------------------------------------------------------------

    /// Element-wise square root.
    pub fn sqrt(&self)  -> Self { self.map_internal(|x| x.sqrt()) }
    /// Element-wise cube root.
    pub fn cbrt(&self)  -> Self { self.map_internal(|x| x.cbrt()) }
    /// Element-wise sine.
    pub fn sin(&self)   -> Self { self.map_internal(|x| x.sin()) }
    /// Element-wise cosine.
    pub fn cos(&self)   -> Self { self.map_internal(|x| x.cos()) }
    /// Element-wise tangent.
    pub fn tan(&self)   -> Self { self.map_internal(|x| x.tan()) }
    pub fn asin(&self)  -> Self { self.map_internal(|x| x.asin()) }
    pub fn acos(&self)  -> Self { self.map_internal(|x| x.acos()) }
    /// Element-wise arctangent.
    pub fn atan(&self)  -> Self { self.map_internal(|x| x.atan()) }
    pub fn sinh(&self)  -> Self { self.map_internal(|x| x.sinh()) }
    pub fn cosh(&self)  -> Self { self.map_internal(|x| x.cosh()) }
    /// Element-wise hyperbolic tangent.
    pub fn tanh(&self)  -> Self { self.map_internal(|x| x.tanh()) }
    /// Element-wise natural exponential: e^x.
    pub fn exp(&self)   -> Self { self.map_internal(|x| x.exp()) }
    pub fn exp2(&self)  -> Self { self.map_internal(|x| x.exp2()) }
    pub fn expm1(&self) -> Self { self.map_internal(|x| x.exp_m1()) }
    /// Element-wise natural logarithm: ln(x).
    pub fn log(&self)   -> Self { self.map_internal(|x| x.ln()) }
    pub fn log2(&self)  -> Self { self.map_internal(|x| x.log2()) }
    pub fn log10(&self) -> Self { self.map_internal(|x| x.log10()) }
    pub fn log1p(&self) -> Self { self.map_internal(|x| x.ln_1p()) }
    pub fn abs(&self)   -> Self { self.map_internal(|x| x.abs()) }
    /// Element-wise ceiling (round toward +inf).
    pub fn ceil(&self)  -> Self { self.map_internal(|x| x.ceil()) }
    /// Element-wise floor (round toward -inf).
    pub fn floor(&self) -> Self { self.map_internal(|x| x.floor()) }
    /// Element-wise rounding to nearest integer.
    pub fn round(&self) -> Self { self.map_internal(|x| x.round()) }
    /// Element-wise truncation toward zero.
    pub fn trunc(&self) -> Self { self.map_internal(|x| x.trunc()) }
    pub fn fract(&self) -> Self { self.map_internal(|x| x.fract()) }
    pub fn signum(&self)-> Self { self.map_internal(|x| if x == 0.0 { 0.0 } else { x.signum() }) }
    pub fn recip(&self) -> Self { self.map_internal(|x| x.recip()) }

    /// Apply the sigmoid function: 1 / (1 + e^-x).
    pub fn sigmoid(&self) -> Self { self.map_internal(|x| 1.0 / (1.0 + (-x).exp())) }
    pub fn clip(&self, lo: f64, hi: f64) -> Self { self.map_internal(|x| x.clamp(lo, hi)) }
    pub fn pow_scalar(&self, exp: f64)   -> Self { self.map_internal(|x| x.powf(exp)) }
    pub fn clamp(&self, lo: f64, hi: f64)-> Self { self.clip(lo, hi) }
    /// Element-wise hypotenuse: sqrt(self² + y²).
    pub fn hypot_scalar(&self, y: f64)   -> Self { self.map_internal(|x| x.hypot(y)) }
    /// Element-wise two-argument arctangent: atan2(self, x).
    pub fn atan2_scalar(&self, x: f64)   -> Self { self.map_internal(|y| y.atan2(x)) }
    /// Element-wise linear interpolation between `self` and `other`.
    pub fn lerp_scalar(&self, other: f64, t: f64) -> Self {
        self.map_internal(|x| x + t * (other - x))
    }
}

// ---------------------------------------------------------------------------
// VectorIter — enables `for x in v:` from Python
//
// IterSource separates the two storage cases so that:
//   Heap vectors  → Arc::clone in __iter__ (O(1), no memcopy, no allocation)
//   Inline vectors → copy ≤32 f64 = ≤256 bytes in __iter__ (unavoidable)
//
// __next__ yields plain f64 → Python float.
// Yielding Scalar would cost one additional PyObject allocation per element
// (~170 ns × N), which on a 100k-element vector adds ~17 ms for nothing.
// Users who need Scalar can wrap: sc.Scalar(x)
// ---------------------------------------------------------------------------

// Not a #[pyclass] — internal enum, lives inside VectorIter only.
enum IterSource {
    Inline([f64; INLINE_MAX], usize),
    Heap(Arc<Vec<f64>>),
}

#[pyclass]
pub struct VectorIter {
    source: IterSource,
    pos: usize,
}

#[pymethods]
impl VectorIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self) -> Option<f64> {
        let val = match &self.source {
            IterSource::Inline(buf, n) => {
                if self.pos >= *n { return None; }
                buf[self.pos]
            }
            IterSource::Heap(arc) => {
                if self.pos >= arc.len() { return None; }
                arc[self.pos]
            }
        };
        self.pos += 1;
        Some(val)
    }
}
use super::core::{Array, PAR_THRESHOLD};
use crate::vector::Vector;
use pyo3::prelude::*;
use rayon::prelude::*;
#[pymethods]
impl Array {
    /// Start a lazy operation pipeline for loop fusion
    pub fn lazy(&self) -> super::lazy::LazyArray {
        super::lazy::LazyArray::new_from_array(self.clone())
    }

    // ── Arithmetic operators ──────────────────────────────────────────────

    /// Element-wise addition.
    ///
    /// Supports broadcasting between Arrays, and scalar-Array addition.
    pub fn __add__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = rhs.py();
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            let this = self.clone();
            let other_arr = other.clone();
            return py.allow_threads(move || this.broadcast_op(&other_arr, |a, b| a + b, "+"));
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            let this = self.clone();
            let v_owned = v.clone();
            return py.allow_threads(move || this.broadcast_vector(&v_owned, |a, b| a + b));
        }
        if let Ok(s) = rhs.extract::<f64>() {
            let this = self.clone();
            return py.allow_threads(move || Ok(this.map_elements(|x| x + s)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for +"))
    }

    pub fn __radd__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    pub fn add_array(&self, other: &Array) -> PyResult<Self> {
        self.broadcast_op(other, |a, b| a + b, "+")
    }

    pub fn sub_array(&self, other: &Array) -> PyResult<Self> {
        self.broadcast_op(other, |a, b| a - b, "-")
    }

    pub fn mul_array_elementwise(&self, other: &Array) -> PyResult<Self> {
        self.broadcast_op(other, |a, b| a * b, "*")
    }

    pub fn div_array_elementwise(&self, other: &Array) -> PyResult<Self> {
        self.broadcast_op(other, |a, b| a / b, "/")
    }

    /// Element-wise subtraction.
    ///
    /// Supports broadcasting between Arrays, and scalar-Array subtraction.
    pub fn __sub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = rhs.py();
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            let this = self.clone();
            let other_arr = other.clone();
            return py.allow_threads(move || this.broadcast_op(&other_arr, |a, b| a - b, "-"));
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            let this = self.clone();
            let v_owned = v.clone();
            return py.allow_threads(move || this.broadcast_vector(&v_owned, |a, b| a - b));
        }
        if let Ok(s) = rhs.extract::<f64>() {
            let this = self.clone();
            return py.allow_threads(move || Ok(this.map_elements(|x| x - s)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for -"))
    }

    pub fn __rsub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            return Ok(self.map_elements(|x| s - x));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Invalid type for rsub",
        ))
    }

    /// Element-wise multiplication.
    ///
    /// Supports broadcasting between Arrays, and scalar-Array multiplication.
    pub fn __mul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = rhs.py();
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            let this = self.clone();
            let other_arr = other.clone();
            return py.allow_threads(move || this.broadcast_op(&other_arr, |a, b| a * b, "*"));
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            let this = self.clone();
            let v_owned = v.clone();
            return py.allow_threads(move || this.broadcast_vector(&v_owned, |a, b| a * b));
        }
        if let Ok(s) = rhs.extract::<f64>() {
            let this = self.clone();
            return py.allow_threads(move || Ok(this.map_elements(|x| x * s)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for *"))
    }

    pub fn __rmul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    /// Element-wise division.
    ///
    /// Supports broadcasting between Arrays, and scalar-Array division.
    /// Uses IEEE-754 semantics for 0-divisors.
    pub fn __truediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            return self.broadcast_op(&other, |a, b| a / b, "/");
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            return self.broadcast_vector(&v, |a, b| a / b);
        }
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "div by zero",
                ));
            }
            return Ok(self.map_elements(|x| x / s));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for /"))
    }

    pub fn __rtruediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            return Ok(self.map_elements(|x| s / x));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Invalid type for rtruediv",
        ))
    }

    /// Element-wise power.
    ///
    /// Supports scalar exponents and Array-Array element-wise power.
    pub fn __pow__<'py>(
        &self,
        rhs: &Bound<'py, PyAny>,
        _modulo: &Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            return self.broadcast_op(&other, |a, b| a.powf(b), "**");
        }
        if let Ok(s) = rhs.extract::<f64>() {
            // Fast-path for common integer exponents
            if s == 2.0 {
                return Ok(self.map_elements(|x| x * x));
            }
            if s == 3.0 {
                return Ok(self.map_elements(|x| x * x * x));
            }
            if s == 0.5 {
                return Ok(self.map_elements(|x| x.sqrt()));
            }
            if s == -1.0 {
                return Ok(self.map_elements(|x| x.recip()));
            }
            if s == 1.0 {
                return Ok(self.clone());
            }
            if s == 0.0 {
                return Ok(Self::from_flat(vec![1.0; self.len()], self.shape.clone()));
            }
            // Check if s is a small positive integer for powi
            if s == s.trunc() && s.abs() <= 32.0 {
                let si = s as i32;
                return Ok(self.map_elements(move |x| x.powi(si)));
            }
            return Ok(self.map_elements(|x| x.powf(s)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Invalid type for **",
        ))
    }

    pub fn pow<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = rhs.py();
        self.__pow__(rhs, &py.None().into_bound(py))
    }

    pub fn __neg__(&self) -> Self {
        self.map_elements(|x| -x)
    }
    pub fn __pos__(&self) -> Self {
        self.clone()
    }
    pub fn __abs__(&self) -> Self {
        self.map_elements(|x| x.abs())
    }

    pub fn add<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__add__(rhs)
    }
    pub fn sub<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__sub__(rhs)
    }
    pub fn mul<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__mul__(rhs)
    }
    pub fn div<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__truediv__(rhs)
    }

    // In-place operators

    pub fn __iadd__<'py>(&mut self, rhs: &Bound<'py, PyAny>) -> PyResult<()> {
        self.inplace_scalar_op(rhs, |a, b| a + b)
    }
    pub fn __isub__<'py>(&mut self, rhs: &Bound<'py, PyAny>) -> PyResult<()> {
        self.inplace_scalar_op(rhs, |a, b| a - b)
    }
    pub fn __imul__<'py>(&mut self, rhs: &Bound<'py, PyAny>) -> PyResult<()> {
        self.inplace_scalar_op(rhs, |a, b| a * b)
    }
    pub fn __itruediv__<'py>(&mut self, rhs: &Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "div by zero",
                ));
            }
        }
        self.inplace_scalar_op(rhs, |a, b| a / b)
    }

    // ── Matrix multiply ───────────────────────────────────────────────────

    pub fn __matmul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.matmul(rhs)
    }

    /// Matrix multiplication (dot product).
    ///
    /// Supports 2D matrix multiplication and vector-matrix multiplication.
    /// Operates outside the Python GIL using the `faer` and `matrixmultiply` backends.
    ///
    /// Examples:
    ///     >>> import rmath.array as ra
    ///     >>> a = ra.Array([[1, 2], [3, 4]])
    ///     >>> b = ra.Array([[2, 0], [1, 2]])
    ///     >>> a.matmul(b)
    ///     Array([[4.0, 4.0], [10.0, 8.0]])
    pub fn matmul<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            let this = self.clone();
            let other_arr = other.clone();
            let res = py.allow_threads(move || this.matmul_array(&other_arr));
            return Ok(res.into_pyobject(py)?.into_any());
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if self.ncols() != v.len_internal() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "matmul dim mismatch",
                ));
            }
            let this = self.clone();
            let v_owned = v.clone();
            let res = py.allow_threads(move || this.matmul_vec(&v_owned));
            return Ok(Vector::new(res).into_pyobject(py)?.into_any());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected Array or Vector",
        ))
    }

    pub fn matmul_trans(&self, v: &Vector) -> PyResult<Vector> {
        if self.nrows() != v.len_internal() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dim mismatch for Transpose Matmul",
            ));
        }
        let (r, c) = (self.nrows(), self.ncols());
        let contig = self.to_contiguous();
        let d = contig.data();
        let result: Vec<f64> = v.with_slice(|s| {
            (0..c)
                .into_par_iter()
                .map(|j| (0..r).map(|i| d[i * c + j] * s[i]).sum())
                .collect()
        });
        Ok(Vector::new(result))
    }

    // ── Elementwise math ──────────────────────────────────────────────────

    pub fn abs(&self) -> Self {
        self.map_elements(|x| x.abs())
    }
    pub fn sqrt(&self) -> Self {
        self.map_elements(|x| x.sqrt())
    }
    pub fn cbrt(&self) -> Self {
        self.map_elements(|x| x.cbrt())
    }
    /// Element-wise exponential.
    ///
    /// Example:
    ///     >>> a = ra.Array([0.0, 1.0])
    ///     >>> a.exp()
    ///     Array([1.0000, 2.7183])
    pub fn exp(&self) -> Self {
        self.map_elements(|x| x.exp())
    }
    pub fn exp2(&self) -> Self {
        self.map_elements(|x| x.exp2())
    }
    pub fn expm1(&self) -> Self {
        self.map_elements(|x| x.exp_m1())
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Self {
        self.map_elements(|x| x.ln())
    }
    pub fn log2(&self) -> Self {
        self.map_elements(|x| x.log2())
    }
    pub fn log10(&self) -> Self {
        self.map_elements(|x| x.log10())
    }
    pub fn log1p(&self) -> Self {
        self.map_elements(|x| x.ln_1p())
    }
    /// Element-wise Sine.
    ///
    /// Example:
    ///     >>> a = ra.Array([0.0, 3.14159 / 2])
    ///     >>> a.sin()
    ///     Array([0.0000, 1.0000])
    pub fn sin(&self) -> Self {
        self.map_elements(|x| x.sin())
    }
    pub fn cos(&self) -> Self {
        self.map_elements(|x| x.cos())
    }
    pub fn tan(&self) -> Self {
        self.map_elements(|x| x.tan())
    }
    pub fn asin(&self) -> Self {
        self.map_elements(|x| x.asin())
    }
    pub fn acos(&self) -> Self {
        self.map_elements(|x| x.acos())
    }
    pub fn atan(&self) -> Self {
        self.map_elements(|x| x.atan())
    }
    pub fn sinh(&self) -> Self {
        self.map_elements(|x| x.sinh())
    }
    pub fn cosh(&self) -> Self {
        self.map_elements(|x| x.cosh())
    }
    pub fn tanh(&self) -> Self {
        self.map_elements(|x| x.tanh())
    }
    pub fn ceil(&self) -> Self {
        self.map_elements(|x| x.ceil())
    }
    pub fn floor(&self) -> Self {
        self.map_elements(|x| x.floor())
    }
    pub fn round(&self) -> Self {
        self.map_elements(|x| x.round())
    }
    pub fn trunc(&self) -> Self {
        self.map_elements(|x| x.trunc())
    }
    pub fn fract(&self) -> Self {
        self.map_elements(|x| x.fract())
    }
    pub fn signum(&self) -> Self {
        self.map_elements(|x| x.signum())
    }
    pub fn recip(&self) -> Self {
        self.map_elements(|x| x.recip())
    }

    pub fn pow_scalar(&self, p: f64) -> Self {
        self.map_elements(|x| x.powf(p))
    }
    pub fn clamp(&self, lo: f64, hi: f64) -> Self {
        self.map_elements(|x| x.clamp(lo, hi))
    }
    pub fn hypot_scalar(&self, other: f64) -> Self {
        self.map_elements(|x| x.hypot(other))
    }
    pub fn atan2_scalar(&self, other: f64) -> Self {
        self.map_elements(|x| x.atan2(other))
    }
    pub fn lerp_scalar(&self, other: f64, t: f64) -> Self {
        self.map_elements(|x| x + t * (other - x))
    }
    pub fn fma(&self, mul: f64, add: f64) -> Self {
        self.map_elements(|x| x.mul_add(mul, add))
    }

    // ── Reductions ────────────────────────────────────────────────────────

    /// Calculate the sum of all elements in the array.
    ///
    /// Uses Kahan compensated summation to maintain precision even with millions
    /// of elements.
    pub fn sum_all(&self) -> f64 {
        let contig = self.to_contiguous();
        let s = contig.data();
        if s.is_empty() {
            return 0.0;
        }
        if s.len() >= PAR_THRESHOLD {
            // Parallel: chunk → Kahan per chunk → Kahan-merge
            let chunk_size = (s.len() / rayon::current_num_threads()).max(1024);
            let partials: Vec<(f64, f64)> = s
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut sum = 0.0_f64;
                    let mut comp = 0.0_f64;
                    for &x in chunk {
                        let y = x - comp;
                        let t = sum + y;
                        comp = (t - sum) - y;
                        sum = t;
                    }
                    (sum, comp)
                })
                .collect();
            let mut total = 0.0_f64;
            let mut comp = 0.0_f64;
            for (ps, pc) in partials {
                // merge partial sum (subtract its residual compensation)
                let y = (ps - pc) - comp;
                let t = total + y;
                comp = (t - total) - y;
                total = t;
            }
            total
        } else {
            let mut sum = 0.0_f64;
            let mut comp = 0.0_f64;
            for &x in s.iter() {
                let y = x - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
            sum
        }
    }
    pub fn prod(&self) -> f64 {
        let contig = self.to_contiguous();
        let d = contig.data();
        if d.is_empty() {
            return 1.0;
        }
        if d.len() >= 131072 {
            d.par_iter().cloned().product()
        } else {
            d.iter().cloned().product()
        }
    }
    /// Calculate the arithmetic mean of all elements in the array.
    pub fn mean(&self) -> f64 {
        let n = self.len();
        if n == 0 {
            return f64::NAN;
        }
        self.sum_all() / n as f64
    }
    /// Calculate the sample variance of all elements in the array.
    ///
    /// Uses Welford's single-pass algorithm for numerical stability.
    pub fn variance(&self) -> f64 {
        let n = self.len();
        if n < 2 {
            return f64::NAN;
        }
        let contig = self.to_contiguous();
        let sd = contig.data();
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        for (i, &x) in sd.iter().enumerate() {
            let delta = x - mean;
            mean += delta / (i + 1) as f64;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }
        m2 / (n - 1) as f64
    }
    /// Calculate the sample standard deviation of all elements in the array.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    /// Find the minimum element in the array.
    pub fn min(&self) -> f64 {
        let contig = self.to_contiguous();
        let d = contig.data();
        if d.is_empty() {
            return f64::INFINITY;
        }
        if d.len() >= 131072 {
            d.par_iter()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        } else {
            d.iter().cloned().fold(f64::INFINITY, f64::min)
        }
    }
    /// Find the maximum element in the array.
    pub fn max(&self) -> f64 {
        let contig = self.to_contiguous();
        let d = contig.data();
        if d.is_empty() {
            return f64::NEG_INFINITY;
        }
        if d.len() >= 131072 {
            d.par_iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        } else {
            d.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    }
    pub fn argmin(&self) -> usize {
        let contig = self.to_contiguous();
        let d = contig.data();
        if d.is_empty() {
            return 0;
        }
        if d.len() >= 131072 {
            d.par_iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        } else {
            d.iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        }
    }
    pub fn argmax(&self) -> usize {
        let contig = self.to_contiguous();
        let d = contig.data();
        if d.is_empty() {
            return 0;
        }
        if d.len() >= 131072 {
            d.par_iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        } else {
            d.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        }
    }

    /// Compute the sum of array elements over a given axis.
    ///
    /// If `axis` is None, sums all elements using Kahan compensated summation.
    /// If `axis` is 0, computes the sum of each column (returns a Vector).
    /// If `axis` is 1, computes the sum of each row (returns a Vector).
    ///
    /// Examples:
    ///     >>> import rmath.array as ra
    ///     >>> a = ra.Array([[1, 2], [3, 4]])
    ///     >>> a.sum(axis=0)
    ///     Vector([4.0, 6.0])
    #[pyo3(signature = (axis=None))]
    pub fn sum<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            None => Ok(self.sum_all().into_pyobject(py)?.into_any()),
            Some(0) => {
                let (r, c) = (self.nrows(), self.ncols());
                let contig = self.to_contiguous();
                let d = contig.data();
                // Cache-friendly: iterate row-major, accumulate into column sums
                let mut sums = vec![0.0f64; c];
                for i in 0..r {
                    let row = &d[i * c..(i + 1) * c];
                    for j in 0..c {
                        sums[j] += row[j];
                    }
                }
                Ok(Vector::new(sums).into_pyobject(py)?.into_any())
            }
            Some(1) => {
                let (r, c) = (self.nrows(), self.ncols());
                let contig = self.to_contiguous();
                let d = contig.data();
                let sums: Vec<f64> = (0..r).map(|i| d[i * c..(i + 1) * c].iter().sum()).collect();
                Ok(Vector::new(sums).into_pyobject(py)?.into_any())
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "axis must be 0 or 1",
            )),
        }
    }

    pub fn mean_axis0(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if r == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Empty array"));
        }
        let contig = self.to_contiguous();
        let d = contig.data();
        // Cache-friendly: iterate row-major
        let mut sums = vec![0.0f64; c];
        for i in 0..r {
            let row = &d[i * c..(i + 1) * c];
            for j in 0..c {
                sums[j] += row[j];
            }
        }
        let inv = 1.0 / r as f64;
        for s in sums.iter_mut() {
            *s *= inv;
        }
        Ok(Vector::new(sums))
    }

    pub fn mean_axis1(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if c == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Empty array"));
        }
        let contig = self.to_contiguous();
        let d = contig.data();
        let sums: Vec<f64> = (0..r)
            .map(|i| d[i * c..(i + 1) * c].iter().sum::<f64>() / c as f64)
            .collect();
        Ok(Vector::new(sums))
    }

    pub fn std_axis0(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if r < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Need at least 2 rows",
            ));
        }
        let mu = self.mean_axis0()?;
        let contig = self.to_contiguous();
        let d = contig.data();
        let stds: Vec<f64> = mu.with_slice(|s| {
            (0..c)
                .map(|j| {
                    let var: f64 =
                        (0..r).map(|i| (d[i * c + j] - s[j]).powi(2)).sum::<f64>() / (r - 1) as f64;
                    var.sqrt() + 1e-8
                })
                .collect()
        });
        Ok(Vector::new(stds))
    }

    pub fn var_axis0(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if r < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Need at least 2 rows",
            ));
        }
        let contig = self.to_contiguous();
        let d = contig.data();

        // Parallel Welford for each column
        let vars: Vec<f64> = (0..c)
            .into_par_iter()
            .map(|j| {
                let mut mean = 0.0;
                let mut m2 = 0.0;
                for i in 0..r {
                    let x = d[i * c + j];
                    let delta = x - mean;
                    mean += delta / (i + 1) as f64;
                    m2 += delta * (x - mean);
                }
                m2 / (r - 1) as f64
            })
            .collect();

        Ok(Vector::new(vars))
    }
    // ── Comparison / mask ops ─────────────────────────────────────────────

    pub fn gt(&self, val: f64) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|&x| x > val)
            .collect()
    }
    pub fn lt(&self, val: f64) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|&x| x < val)
            .collect()
    }
    pub fn ge(&self, val: f64) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|&x| x >= val)
            .collect()
    }
    pub fn le(&self, val: f64) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|&x| x <= val)
            .collect()
    }
    pub fn eq_scalar(&self, val: f64) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|&x| x == val)
            .collect()
    }
    pub fn ne_scalar(&self, val: f64) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|&x| x != val)
            .collect()
    }
    pub fn isnan(&self) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|x| x.is_nan())
            .collect()
    }
    pub fn isfinite(&self) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|x| x.is_finite())
            .collect()
    }
    pub fn isinf(&self) -> Vec<bool> {
        self.to_contiguous()
            .data()
            .iter()
            .map(|x| x.is_infinite())
            .collect()
    }

    pub fn where_scalar(&self, mask: Vec<bool>, other: f64) -> PyResult<Self> {
        if mask.len() != self.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Mask len mismatch"));
        }
        let contig = self.to_contiguous();
        let data: Vec<f64> = contig
            .data()
            .iter()
            .zip(mask.iter())
            .map(|(&x, &m)| if m { x } else { other })
            .collect();
        Ok(Self::from_flat(data, self.shape.clone()))
    }

    /// Compute the Pearson correlation matrix (rows = variables).
    ///
    /// Returns an N x N correlation matrix where element (i, j) is the
    /// correlation between row i and row j.
    pub fn correlation_matrix<'py>(&self, py: Python<'py>) -> PyResult<Self> {
        crate::stats::inferential::correlation_matrix(py, self)
    }

    /// Compute the pairwise Euclidean distance matrix between rows of two arrays.
    ///
    /// Returns an M x N array of distances where M is self.nrows() and N is other.nrows().
    pub fn cdist<'py>(&self, py: Python<'py>, other: &Array) -> PyResult<Self> {
        crate::geometry::cdist(py, self, other)
    }

    /// Calculate the Euclidean distance between this Array (rows) and a query point.
    /// If the query is a 1D Vector, returns a 1D Vector of distances.
    pub fn distance<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(v) = query.extract::<PyRef<Vector>>() {
            let (r, c) = (self.nrows(), self.ncols());
            if c != v.len_internal() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Dimension mismatch between Array columns and query Vector",
                ));
            }
            let contig = self.to_contiguous();
            let d = contig.data();
            let distances: Vec<f64> = v.with_slice(|vs| {
                if r >= PAR_THRESHOLD {
                    (0..r)
                        .into_par_iter()
                        .map(|i| {
                            let row = &d[i * c..(i + 1) * c];
                            row.iter()
                                .zip(vs.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt()
                        })
                        .collect()
                } else {
                    (0..r)
                        .map(|i| {
                            let row = &d[i * c..(i + 1) * c];
                            row.iter()
                                .zip(vs.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt()
                        })
                        .collect()
                }
            });
            Ok(Vector::new(distances).into_pyobject(py)?.into_any())
        } else if let Ok(arr) = query.extract::<PyRef<Array>>() {
            // Fall back to cdist if passing an Array
            let res = crate::geometry::cdist(py, self, &arr)?;
            Ok(res.into_pyobject(py)?.into_any())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Query must be a Vector or Array",
            ))
        }
    }
}

// ── Module-level Functional API ───────────────────────────────────────────

#[pyfunction]
pub fn arange(start: f64, stop: f64, step: f64) -> PyResult<Array> {
    Array::arange(start, stop, step)
}

#[pyfunction(name = "range")]
pub fn array_range(start: f64, stop: f64, step: f64) -> PyResult<Array> {
    Array::arange(start, stop, step)
}

/// Create a new Array of zeros with given rows and columns.
///
/// Example:
///     >>> ra.zeros(2, 3)
///     Array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
#[pyfunction]
pub fn zeros(rows: usize, cols: usize) -> Array {
    Array::zeros_internal(&[rows, cols])
}

/// Create a new Array of ones with given rows and columns.
///
/// Example:
///     >>> ra.ones(2, 2)
///     Array([[1.0, 1.0], [1.0, 1.0]])
#[pyfunction]
pub fn ones(rows: usize, cols: usize) -> Array {
    Array::ones_internal(&[rows, cols])
}

/// Create a new Array with samples from the standard normal distribution.
///
/// Example:
///     >>> ra.randn(100, 100) # Create 100x100 weight matrix
#[pyfunction(signature = (*shape))]
pub fn randn(shape: Vec<usize>) -> Array {
    Array::randn(shape)
}

#[pyfunction(signature = (*shape))]
pub fn rand_uniform(shape: Vec<usize>) -> Array {
    Array::rand_uniform(shape)
}

// ── Internal non-pymethods ────────────────────────────────────────────────────

impl Array {
    fn broadcast_op(
        &self,
        other: &Array,
        f: impl Fn(f64, f64) -> f64 + Sync + Copy,
        _op: &str,
    ) -> PyResult<Self> {
        let res_shape = Array::broadcast_shapes(&self.shape, &other.shape)?;

        // Tier 1: Identical Shapes (Hot Path)
        if self.shape == other.shape && self.is_contiguous() && other.is_contiguous() {
            let s_data = self.data();
            let o_data = other.data();
            let n = s_data.len();
            let data: Vec<f64> = if n >= PAR_THRESHOLD {
                s_data
                    .par_iter()
                    .zip(o_data.par_iter())
                    .map(|(&a, &b)| f(a, b))
                    .collect()
            } else {
                s_data
                    .iter()
                    .zip(o_data.iter())
                    .map(|(&a, &b)| f(a, b))
                    .collect()
            };
            return Ok(Self::from_flat(data, self.shape.clone()));
        }

        // Tier 2: 2D Broadcasting Fast-Path (Parallelized)
        if self.ndim() <= 2 && other.ndim() <= 2 && self.is_contiguous() && other.is_contiguous() {
            fn get_rc(s: &[usize]) -> (usize, usize) {
                match s.len() {
                    0 => (1, 1),
                    1 => (1, s[0]), // [N] -> [1, N]
                    _ => (s[0], s[1]),
                }
            }
            let (r1, c1) = get_rc(&self.shape);
            let (r2, c2) = get_rc(&other.shape);
            let rout = res_shape[0];
            let cout = if res_shape.len() > 1 { res_shape[1] } else { 1 };

            let d1 = self.data();
            let d2 = other.data();
            let total = rout * cout;

            if total >= PAR_THRESHOLD {
                // Parallel: use a flat iterator to avoid inner row allocations
                let out: Vec<f64> = (0..total)
                    .into_par_iter()
                    .map(|k| {
                        let i = k / cout;
                        let j = k % cout;
                        let row1_idx = if r1 == 1 { 0 } else { i * c1 };
                        let row2_idx = if r2 == 1 { 0 } else { i * c2 };
                        let v1 = d1[row1_idx + if c1 == 1 { 0 } else { j }];
                        let v2 = d2[row2_idx + if c2 == 1 { 0 } else { j }];
                        f(v1, v2)
                    })
                    .collect();
                return Ok(Self::from_flat(out, res_shape));
            } else {
                let mut out = Vec::with_capacity(total);
                for i in 0..rout {
                    let row1_idx = if r1 == 1 { 0 } else { i * c1 };
                    let row2_idx = if r2 == 1 { 0 } else { i * c2 };
                    for j in 0..cout {
                        let v1 = d1[row1_idx + if c1 == 1 { 0 } else { j }];
                        let v2 = d2[row2_idx + if c2 == 1 { 0 } else { j }];
                        out.push(f(v1, v2));
                    }
                }
                return Ok(Self::from_flat(out, res_shape));
            }
        }

        // Tier 3: General N-D Broadcasting (Stride-Aware, Parallelized)
        let n: usize = res_shape.iter().product();
        let res_strides = Array::compute_strides(&res_shape);
        let d1 = self.storage_slice();
        let d2 = other.storage_slice();
        let s1 = &self.shape;
        let s2 = &other.shape;
        let st1 = &self.strides;
        let st2 = &other.strides;
        let res_ndim = res_shape.len();
        let off1 = self.offset as isize;
        let off2 = other.offset as isize;

        let compute_element = |i: usize| -> f64 {
            let mut idx1 = off1;
            let mut idx2 = off2;
            let mut rem = i;
            for d in 0..res_ndim {
                let res_stride_d = res_strides[d] as usize;
                let coord = rem / res_stride_d;
                rem %= res_stride_d;
                if d >= res_ndim - s1.len() {
                    let s_d = d - (res_ndim - s1.len());
                    if s1[s_d] > 1 {
                        idx1 += coord as isize * st1[s_d];
                    }
                }
                if d >= res_ndim - s2.len() {
                    let s_d = d - (res_ndim - s2.len());
                    if s2[s_d] > 1 {
                        idx2 += coord as isize * st2[s_d];
                    }
                }
            }
            f(d1[idx1 as usize], d2[idx2 as usize])
        };

        let out_data: Vec<f64> = if n >= PAR_THRESHOLD {
            (0..n).into_par_iter().map(compute_element).collect()
        } else {
            (0..n).map(compute_element).collect()
        };

        Ok(Self::from_flat(out_data, res_shape))
    }

    fn broadcast_vector(
        &self,
        v: &Vector,
        f: impl Fn(f64, f64) -> f64 + Sync + Send + Copy,
    ) -> PyResult<Self> {
        if self.ndim() < 2 || v.len_internal() != self.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vector broadcast len mismatch",
            ));
        }
        let (r, c) = (self.nrows(), self.ncols());
        let contig = self.to_contiguous();
        let sd = contig.data();
        let n = r * c;
        let data: Vec<f64> = v.with_slice(|vs| {
            if n >= PAR_THRESHOLD {
                (0..n)
                    .into_par_iter()
                    .map(|k| f(sd[k], vs[k % c]))
                    .collect()
            } else {
                (0..n).map(|k| f(sd[k], vs[k % c])).collect()
            }
        });
        Ok(Self::from_flat(data, vec![r, c]))
    }

    fn inplace_scalar_op<'py>(
        &mut self,
        rhs: &Bound<'py, PyAny>,
        f: impl Fn(f64, f64) -> f64,
    ) -> PyResult<()> {
        if let Ok(s) = rhs.extract::<f64>() {
            self.make_owned();
            let n = self.len();
            match &mut self.storage {
                super::core::ArrayStorage::Inline(d, _) => {
                    for i in 0..n {
                        d[i] = f(d[i], s);
                    }
                }
                super::core::ArrayStorage::Heap(arc) => {
                    let v = std::sync::Arc::get_mut(arc).unwrap();
                    for x in v.iter_mut() {
                        *x = f(*x, s);
                    }
                }
            }
            return Ok(());
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if v.len_internal() != self.ncols() {
                return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch"));
            }
            self.make_owned();
            let (r, c) = (self.nrows(), self.ncols());
            v.with_slice(|vs| match &mut self.storage {
                super::core::ArrayStorage::Inline(d, _) => {
                    for i in 0..r {
                        for j in 0..c {
                            d[i * c + j] = f(d[i * c + j], vs[j]);
                        }
                    }
                }
                super::core::ArrayStorage::Heap(arc) => {
                    let data = std::sync::Arc::get_mut(arc).unwrap();
                    for i in 0..r {
                        for j in 0..c {
                            data[i * c + j] = f(data[i * c + j], vs[j]);
                        }
                    }
                }
            });
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected float or Vector",
        ))
    }

    pub fn matmul_array(&self, other: &Array) -> Array {
        let nd1 = self.ndim();
        let nd2 = other.ndim();

        // 2D Fast Path using matrixmultiply (Zero-copy layout aware)
        if nd1 == 2 && nd2 == 2 {
            let m = self.nrows();
            let k = self.ncols();
            let n = other.ncols();
            if k != other.nrows() {
                panic!("MatMul dim mismatch: {}x{} @ {}x{}", m, k, other.nrows(), n);
            }

            let mut out = vec![0.0; m * n];
            let rsa = self.strides[0];
            let csa = self.strides[1];
            let rsb = other.strides[0];
            let csb = other.strides[1];

            unsafe {
                matrixmultiply::dgemm(
                    m,
                    k,
                    n,
                    1.0,
                    self.storage_slice()[self.offset..].as_ptr(),
                    rsa,
                    csa,
                    other.storage_slice()[other.offset..].as_ptr(),
                    rsb,
                    csb,
                    0.0,
                    out.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
            return Self::from_flat(out, vec![m, n]);
        }

        // Batch MatMul (Rank 3+)
        let batch_shape1 = if nd1 > 2 { &self.shape[..nd1 - 2] } else { &[] };
        let batch_shape2 = if nd2 > 2 {
            &other.shape[..nd2 - 2]
        } else {
            &[]
        };

        let res_batch = Array::broadcast_shapes(batch_shape1, batch_shape2).unwrap();
        let m = self.shape[nd1 - 2];
        let k = self.shape[nd1 - 1];
        let n = other.shape[nd2 - 1];
        // Note: inner dim k matching is checked in the 2D path or already verified

        let mut res_shape = res_batch.clone();
        res_shape.push(m);
        res_shape.push(n);

        let batch_count: usize = res_batch.iter().product();
        let slice_size = m * n;
        let mut out_data = vec![0.0; batch_count * slice_size];

        // Strides for the 2D MatMul part
        let rsa = self.strides[nd1 - 2];
        let csa = self.strides[nd1 - 1];
        let rsb = other.strides[nd2 - 2];
        let csb = other.strides[nd2 - 1];

        // Shared slices
        let s_slice = self.storage_slice();
        let o_slice = other.storage_slice();

        // Stride-aware batch walker
        let res_batch_strides = Array::compute_strides(&res_batch);
        let b_st1 = if nd1 > 2 {
            &self.strides[..nd1 - 2]
        } else {
            &[]
        };
        let b_st2 = if nd2 > 2 {
            &other.strides[..nd2 - 2]
        } else {
            &[]
        };

        for i in 0..batch_count {
            // Calculate base offsets for this batch slice
            let mut offset1 = self.offset as isize;
            let mut offset2 = other.offset as isize;
            let mut rem = i;

            for d in 0..res_batch.len() {
                let stride = res_batch_strides[d] as usize;
                let coord = rem / stride;
                rem %= stride;

                // Handle broadcasting for batch dimensions
                if d >= res_batch.len() - batch_shape1.len() {
                    let sd = d - (res_batch.len() - batch_shape1.len());
                    if batch_shape1[sd] > 1 {
                        offset1 += coord as isize * b_st1[sd];
                    }
                }
                if d >= res_batch.len() - batch_shape2.len() {
                    let sd = d - (res_batch.len() - batch_shape2.len());
                    if batch_shape2[sd] > 1 {
                        offset2 += coord as isize * b_st2[sd];
                    }
                }
            }

            // Direct dgemm call on this slice
            unsafe {
                matrixmultiply::dgemm(
                    m,
                    k,
                    n,
                    1.0,
                    s_slice.as_ptr().offset(offset1),
                    rsa,
                    csa,
                    o_slice.as_ptr().offset(offset2),
                    rsb,
                    csb,
                    0.0,
                    out_data.as_mut_ptr().add(i * slice_size),
                    n as isize,
                    1,
                );
            }
        }

        Self::from_flat(out_data, res_shape)
    }

    pub fn matmul_vec(&self, v: &Vector) -> Vec<f64> {
        let (r, c) = (self.nrows(), self.ncols());
        let contig = self.to_contiguous();
        let d = contig.data();
        v.with_slice(|s| {
            (0..r)
                .into_par_iter()
                .map(|i| (0..c).map(|j| d[i * c + j] * s[j]).sum())
                .collect()
        })
    }
}

// ── Internal performance kernels (not exposed to Python) ──────────────────

impl Array {
    /// Fused division backward pass with direct accumulation.
    /// If da_acc or db_acc are provided, results are added directly to them.
    pub fn div_backward_fused(
        &self,
        other: &Array,
        grad: &Array,
        mut da_acc: Option<&mut [f64]>,
        mut db_acc: Option<&mut [f64]>,
    ) -> (Option<Array>, Option<Array>) {
        let n = self.len();

        let a_contig = self.to_contiguous();
        let b_contig = other.to_contiguous();
        let g_contig = grad.to_contiguous();

        let a_data = a_contig.data();
        let b_data = b_contig.data();
        let g_data = g_contig.data();

        if n >= crate::array::core::PAR_THRESHOLD {
            // Parallel fused kernel with True Dual-Buffer Fusion
            if da_acc.is_some() && db_acc.is_some() {
                let da = da_acc.unwrap();
                let db = db_acc.unwrap();
                da.par_iter_mut()
                    .zip(db.par_iter_mut())
                    .zip(
                        a_data
                            .par_iter()
                            .zip(b_data.par_iter().zip(g_data.par_iter())),
                    )
                    .for_each(|((da_out, db_out), (&ai, (&bi, &gi)))| {
                        let r = 1.0 / bi;
                        *da_out += gi * r;
                        *db_out += gi * (-ai * (r * r));
                    });
            } else if let Some(acc) = da_acc {
                acc.par_iter_mut()
                    .zip(b_data.par_iter().zip(g_data.par_iter()))
                    .for_each(|(out, (&bi, &gi))| *out += gi * (1.0 / bi));
            } else if let Some(acc) = db_acc {
                acc.par_iter_mut()
                    .zip(
                        a_data
                            .par_iter()
                            .zip(b_data.par_iter().zip(g_data.par_iter())),
                    )
                    .for_each(|(out, (&ai, (&bi, &gi)))| {
                        let r = 1.0 / bi;
                        *out += gi * (-ai * (r * r));
                    });
            }
        } else {
            // Serial fused kernel
            for i in 0..n {
                let r = 1.0 / b_data[i];
                let gi = g_data[i];
                if let Some(ref mut acc) = da_acc {
                    acc[i] += gi * r;
                }
                if let Some(ref mut acc) = db_acc {
                    acc[i] += gi * (-a_data[i] * (r * r));
                }
            }
        }

        (None, None) // Buffers were accumulated in-place
    }
}

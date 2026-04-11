use pyo3::prelude::*;
use rayon::prelude::*;
use crate::vector::Vector;
use super::core::{Array, PAR_THRESHOLD};
#[pymethods]
impl Array {

    // ── Arithmetic operators ──────────────────────────────────────────────

    pub fn __add__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            return self.broadcast_op(&other, |a, b| a + b, "+");
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            return self.broadcast_vector(&v, |a, b| a + b);
        }
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_elements(|x| x + s)); }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for +"))
    }

    pub fn __radd__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> { self.__add__(rhs) }

    pub fn add_array(&self, other: &Array) -> PyResult<Self> {
        self.broadcast_op(other, |a, b| a + b, "+")
    }

    pub fn sub_array(&self, other: &Array) -> PyResult<Self> {
        self.broadcast_op(other, |a, b| a - b, "-")
    }

    pub fn mul_array_elementwise(&self, other: &Array) -> PyResult<Self> {
        self.broadcast_op(other, |a, b| a * b, "*")
    }

    pub fn __sub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            return self.broadcast_op(&other, |a, b| a - b, "-");
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            return self.broadcast_vector(&v, |a, b| a - b);
        }
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_elements(|x| x - s)); }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for -"))
    }

    pub fn __rsub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_elements(|x| s - x)); }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for rsub"))
    }

    pub fn __mul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            return self.broadcast_op(&other, |a, b| a * b, "*");
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            return self.broadcast_vector(&v, |a, b| a * b);
        }
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_elements(|x| x * s)); }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for *"))
    }

    pub fn __rmul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> { self.__mul__(rhs) }

    pub fn __truediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            return self.broadcast_op(&other, |a, b| a / b, "/");
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            return self.broadcast_vector(&v, |a, b| a / b);
        }
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("div by zero")); }
            return Ok(self.map_elements(|x| x / s));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for /"))
    }

    pub fn __rtruediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() { return Ok(self.map_elements(|x| s / x)); }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for rtruediv"))
    }

    pub fn __pow__<'py>(&self, rhs: &Bound<'py, PyAny>, _modulo: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            return self.broadcast_op(&other, |a, b| a.powf(b), "**");
        }
        if let Ok(s) = rhs.extract::<f64>() {
            // Fast-path for common integer exponents
            if s == 2.0      { return Ok(self.map_elements(|x| x * x)); }
            if s == 3.0      { return Ok(self.map_elements(|x| x * x * x)); }
            if s == 0.5      { return Ok(self.map_elements(|x| x.sqrt())); }
            if s == -1.0     { return Ok(self.map_elements(|x| x.recip())); }
            if s == 1.0      { return Ok(self.clone()); }
            if s == 0.0      { return Ok(Self::from_flat(vec![1.0; self.len()], self.shape.clone())); }
            // Check if s is a small positive integer for powi
            if s == s.trunc() && s.abs() <= 32.0 {
                let si = s as i32;
                return Ok(self.map_elements(move |x| x.powi(si)));
            }
            return Ok(self.map_elements(|x| x.powf(s)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid type for **"))
    }

    pub fn pow<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = rhs.py();
        self.__pow__(rhs, &py.None().into_bound(py))
    }

    pub fn __neg__(&self) -> Self { self.map_elements(|x| -x) }
    pub fn __pos__(&self) -> Self { self.clone() }
    pub fn __abs__(&self) -> Self { self.map_elements(|x| x.abs()) }

    pub fn add<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> { self.__add__(rhs) }
    pub fn sub<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> { self.__sub__(rhs) }
    pub fn mul<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> { self.__mul__(rhs) }
    pub fn div<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> { self.__truediv__(rhs) }

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
            if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("div by zero")); }
        }
        self.inplace_scalar_op(rhs, |a, b| a / b)
    }

    // ── Matrix multiply ───────────────────────────────────────────────────

    pub fn __matmul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            let res = self.matmul_array(&other);
            return Ok(res.into_pyobject(py)?.into_any());
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if self.ncols() != v.len_internal() {
                return Err(pyo3::exceptions::PyValueError::new_err("matmul dim mismatch"));
            }
            let res = self.matmul_vec(&v);
            return Ok(Vector::new(res).into_pyobject(py)?.into_any());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Array or Vector"))
    }

    pub fn matmul_trans(&self, v: &Vector) -> PyResult<Vector> {
        if self.nrows() != v.len_internal() {
            return Err(pyo3::exceptions::PyValueError::new_err("Dim mismatch for Transpose Matmul"));
        }
        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        let result: Vec<f64> = v.with_slice(|s| {
            (0..c).into_par_iter().map(|j| {
                (0..r).map(|i| d[i * c + j] * s[i]).sum()
            }).collect()
        });
        Ok(Vector::new(result))
    }

    // ── Elementwise math ──────────────────────────────────────────────────

    pub fn abs(&self)   -> Self { self.map_elements(|x| x.abs()) }
    pub fn sqrt(&self)  -> Self { self.map_elements(|x| x.sqrt()) }
    pub fn cbrt(&self)  -> Self { self.map_elements(|x| x.cbrt()) }
    pub fn exp(&self)   -> Self { self.map_elements(|x| x.exp()) }
    pub fn exp2(&self)  -> Self { self.map_elements(|x| x.exp2()) }
    pub fn expm1(&self) -> Self { self.map_elements(|x| x.exp_m1()) }
    pub fn log(&self)   -> Self { self.map_elements(|x| x.ln()) }
    pub fn log2(&self)  -> Self { self.map_elements(|x| x.log2()) }
    pub fn log10(&self) -> Self { self.map_elements(|x| x.log10()) }
    pub fn log1p(&self) -> Self { self.map_elements(|x| x.ln_1p()) }
    pub fn sin(&self)   -> Self { self.map_elements(|x| x.sin()) }
    pub fn cos(&self)   -> Self { self.map_elements(|x| x.cos()) }
    pub fn tan(&self)   -> Self { self.map_elements(|x| x.tan()) }
    pub fn asin(&self)  -> Self { self.map_elements(|x| x.asin()) }
    pub fn acos(&self)  -> Self { self.map_elements(|x| x.acos()) }
    pub fn atan(&self)  -> Self { self.map_elements(|x| x.atan()) }
    pub fn sinh(&self)  -> Self { self.map_elements(|x| x.sinh()) }
    pub fn cosh(&self)  -> Self { self.map_elements(|x| x.cosh()) }
    pub fn tanh(&self)  -> Self { self.map_elements(|x| x.tanh()) }
    pub fn ceil(&self)  -> Self { self.map_elements(|x| x.ceil()) }
    pub fn floor(&self) -> Self { self.map_elements(|x| x.floor()) }
    pub fn round(&self) -> Self { self.map_elements(|x| x.round()) }
    pub fn trunc(&self) -> Self { self.map_elements(|x| x.trunc()) }
    pub fn fract(&self) -> Self { self.map_elements(|x| x.fract()) }
    pub fn signum(&self)-> Self { self.map_elements(|x| x.signum()) }
    pub fn recip(&self) -> Self { self.map_elements(|x| x.recip()) }

    pub fn pow_scalar(&self, p: f64) -> Self { self.map_elements(|x| x.powf(p)) }
    pub fn clamp(&self, lo: f64, hi: f64) -> Self { self.map_elements(|x| x.clamp(lo, hi)) }
    pub fn hypot_scalar(&self, other: f64) -> Self { self.map_elements(|x| x.hypot(other)) }
    pub fn atan2_scalar(&self, other: f64) -> Self { self.map_elements(|x| x.atan2(other)) }
    pub fn lerp_scalar(&self, other: f64, t: f64) -> Self {
        self.map_elements(|x| x + t * (other - x))
    }
    pub fn fma(&self, mul: f64, add: f64) -> Self { self.map_elements(|x| x.mul_add(mul, add)) }

    // ── Reductions ────────────────────────────────────────────────────────

    pub fn sum_all(&self) -> f64 {
        let s = self.data();
        if s.len() >= 131072 {
            s.par_iter().sum::<f64>()
        } else {
            s.iter().sum::<f64>()
        }
    }
    pub fn prod(&self)    -> f64 { 
        let d = self.data();
        if d.is_empty() { return 1.0; }
        if d.len() >= 131072 {
            d.par_iter().cloned().product()
        } else {
            d.iter().cloned().product()
        }
    }
    pub fn mean(&self)    -> f64 {
        let n = self.len();
        if n == 0 { return f64::NAN; }
        self.sum_all() / n as f64
    }
    pub fn min(&self) -> f64 {
        let d = self.data();
        if d.is_empty() { return f64::INFINITY; }
        if d.len() >= 131072 {
            d.par_iter().cloned().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        } else {
            d.iter().cloned().fold(f64::INFINITY, f64::min)
        }
    }
    pub fn max(&self) -> f64 {
        let d = self.data();
        if d.is_empty() { return f64::NEG_INFINITY; }
        if d.len() >= 131072 {
            d.par_iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        } else {
            d.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    }
    pub fn argmin(&self) -> usize {
        let d = self.data();
        if d.is_empty() { return 0; }
        if d.len() >= 131072 {
            d.par_iter().enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        } else {
            d.iter().enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        }
    }
    pub fn argmax(&self) -> usize {
        let d = self.data();
        if d.is_empty() { return 0; }
        if d.len() >= 131072 {
            d.par_iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        } else {
            d.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap()
        }
    }

    #[pyo3(signature = (axis=None))]
    pub fn sum<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            None => Ok(self.sum_all().into_pyobject(py)?.into_any()),
            Some(0) => {
                let (r, c) = (self.nrows(), self.ncols());
                let d = self.data();
                // Cache-friendly: iterate row-major, accumulate into column sums
                let mut sums = vec![0.0f64; c];
                for i in 0..r {
                    let row = &d[i*c..(i+1)*c];
                    for j in 0..c { sums[j] += row[j]; }
                }
                Ok(Vector::new(sums).into_pyobject(py)?.into_any())
            }
            Some(1) => {
                let (r, c) = (self.nrows(), self.ncols());
                let d = self.data();
                let sums: Vec<f64> = (0..r).map(|i| d[i*c..(i+1)*c].iter().sum()).collect();
                Ok(Vector::new(sums).into_pyobject(py)?.into_any())
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err("axis must be 0 or 1"))
        }
    }

    pub fn mean_axis0(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if r == 0 { return Err(pyo3::exceptions::PyValueError::new_err("Empty array")); }
        let d = self.data();
        // Cache-friendly: iterate row-major
        let mut sums = vec![0.0f64; c];
        for i in 0..r {
            let row = &d[i*c..(i+1)*c];
            for j in 0..c { sums[j] += row[j]; }
        }
        let inv = 1.0 / r as f64;
        for s in sums.iter_mut() { *s *= inv; }
        Ok(Vector::new(sums))
    }

    pub fn mean_axis1(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if c == 0 { return Err(pyo3::exceptions::PyValueError::new_err("Empty array")); }
        let d = self.data();
        let sums: Vec<f64> = (0..r).map(|i| d[i*c..(i+1)*c].iter().sum::<f64>() / c as f64).collect();
        Ok(Vector::new(sums))
    }

    pub fn std_axis0(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if r < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Need at least 2 rows")); }
        let mu = self.mean_axis0()?;
        let d = self.data();
        let stds: Vec<f64> = mu.with_slice(|s| {
            (0..c).map(|j| {
                let var: f64 = (0..r).map(|i| (d[i*c+j] - s[j]).powi(2)).sum::<f64>() / (r-1) as f64;
                var.sqrt() + 1e-8
            }).collect()
        });
        Ok(Vector::new(stds))
    }

    pub fn var_axis0(&self) -> PyResult<Vector> {
        let (r, c) = (self.nrows(), self.ncols());
        if r < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Need at least 2 rows")); }
        let d = self.data();
        
        // Parallel Welford for each column
        let vars: Vec<f64> = (0..c).into_par_iter().map(|j| {
            let mut mean = 0.0;
            let mut m2 = 0.0;
            for i in 0..r {
                let x = d[i * c + j];
                let delta = x - mean;
                mean += delta / (i + 1) as f64;
                m2 += delta * (x - mean);
            }
            m2 / (r - 1) as f64
        }).collect();
        
        Ok(Vector::new(vars))
    }

    // ── Comparison / mask ops ─────────────────────────────────────────────

    pub fn gt(&self, val: f64)  -> Vec<bool> { self.data().iter().map(|&x| x > val).collect() }
    pub fn lt(&self, val: f64)  -> Vec<bool> { self.data().iter().map(|&x| x < val).collect() }
    pub fn ge(&self, val: f64)  -> Vec<bool> { self.data().iter().map(|&x| x >= val).collect() }
    pub fn le(&self, val: f64)  -> Vec<bool> { self.data().iter().map(|&x| x <= val).collect() }
    pub fn eq_scalar(&self, val: f64) -> Vec<bool> { self.data().iter().map(|&x| x == val).collect() }
    pub fn ne_scalar(&self, val: f64) -> Vec<bool> { self.data().iter().map(|&x| x != val).collect() }
    pub fn isnan(&self)    -> Vec<bool> { self.data().iter().map(|x| x.is_nan()).collect() }
    pub fn isfinite(&self) -> Vec<bool> { self.data().iter().map(|x| x.is_finite()).collect() }
    pub fn isinf(&self)    -> Vec<bool> { self.data().iter().map(|x| x.is_infinite()).collect() }

    pub fn where_scalar(&self, mask: Vec<bool>, other: f64) -> PyResult<Self> {
        if mask.len() != self.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Mask len mismatch"));
        }
        let data: Vec<f64> = self.data().iter().zip(mask.iter())
            .map(|(&x, &m)| if m { x } else { other }).collect();
        Ok(Self::from_flat(data, self.shape.clone()))
    }
}

// ── Internal non-pymethods ────────────────────────────────────────────────────

impl Array {
    fn broadcast_op(&self, other: &Array, f: impl Fn(f64, f64) -> f64 + Sync + Copy, _op: &str) -> PyResult<Self> {
        let res_shape = Array::broadcast_shapes(&self.shape, &other.shape)?;
        
        // Tier 1: Identical Shapes (Hot Path)
        if self.shape == other.shape {
            let s_data = self.data();
            let o_data = other.data();
            let n = s_data.len();
            let data: Vec<f64> = if n >= PAR_THRESHOLD {
                s_data.par_iter().zip(o_data.par_iter()).map(|(&a, &b)| f(a, b)).collect()
            } else {
                s_data.iter().zip(o_data.iter()).map(|(&a, &b)| f(a, b)).collect()
            };
            return Ok(Self::from_flat(data, self.shape.clone()));
        }

        // Tier 2: 2D Broadcasting Fast-Path
        if self.ndim() <= 2 && other.ndim() <= 2 {
            fn get_rc(s: &[usize]) -> (usize, usize) {
                match s.len() {
                    0 => (1, 1),
                    1 => (1, s[0]),  // [N] -> [1, N]
                    _ => (s[0], s[1]),
                }
            }
            let (r1, c1) = get_rc(&self.shape);
            let (r2, c2) = get_rc(&other.shape);
            let rout = res_shape[0];
            let cout = if res_shape.len() > 1 { res_shape[1] } else { 1 };

            let d1 = self.data();
            let d2 = other.data();
            let mut out = Vec::with_capacity(rout * cout);
            
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

        // Tier 3: General N-D Broadcasting (Slow Path)
        let n: usize = res_shape.iter().product();
        let res_strides = Array::compute_strides(&res_shape);
        let d1 = self.data();
        let d2 = other.data();
        let s1 = &self.shape;
        let s2 = &other.shape;
        let st1 = &self.strides;
        let st2 = &other.strides;
        let res_ndim = res_shape.len();

        let out_data: Vec<f64> = (0..n).map(|i| {
            let mut idx1 = 0;
            let mut idx2 = 0;
            let mut rem = i;
            for d in 0..res_ndim {
                let coord = rem / res_strides[d];
                rem %= res_strides[d];
                if d >= res_ndim - s1.len() {
                    let s_d = d - (res_ndim - s1.len());
                    if s1[s_d] > 1 { idx1 += coord * st1[s_d]; }
                }
                if d >= res_ndim - s2.len() {
                    let s_d = d - (res_ndim - s2.len());
                    if s2[s_d] > 1 { idx2 += coord * st2[s_d]; }
                }
            }
            f(d1[idx1], d2[idx2])
        }).collect();
        
        Ok(Self::from_flat(out_data, res_shape))
    }

    fn broadcast_vector(&self, v: &Vector, f: impl Fn(f64,f64)->f64) -> PyResult<Self> {
        if self.ndim() < 2 || v.len_internal() != self.ncols() {
            return Err(pyo3::exceptions::PyValueError::new_err("Vector broadcast len mismatch"));
        }
        let (r, c) = (self.nrows(), self.ncols());
        let sd = self.data();
        let data: Vec<f64> = v.with_slice(|vs| {
            (0..r*c).map(|k| f(sd[k], vs[k % c])).collect()
        });
        Ok(Self::from_flat(data, vec![r, c]))
    }

    fn inplace_scalar_op<'py>(&mut self, rhs: &Bound<'py, PyAny>, f: impl Fn(f64,f64)->f64) -> PyResult<()> {
        if let Ok(s) = rhs.extract::<f64>() {
            self.make_owned();
            let n = self.len();
            match &mut self.storage {
                super::core::ArrayStorage::Inline(d, _) => {
                    for i in 0..n { d[i] = f(d[i], s); }
                }
                super::core::ArrayStorage::Heap(arc) => {
                    let v = std::sync::Arc::get_mut(arc).unwrap();
                    for x in v.iter_mut() { *x = f(*x, s); }
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
            v.with_slice(|vs| {
                match &mut self.storage {
                    super::core::ArrayStorage::Inline(d, _) => {
                        for i in 0..r { for j in 0..c { d[i*c+j] = f(d[i*c+j], vs[j]); } }
                    }
                    super::core::ArrayStorage::Heap(arc) => {
                        let data = std::sync::Arc::get_mut(arc).unwrap();
                        for i in 0..r { for j in 0..c { data[i*c+j] = f(data[i*c+j], vs[j]); } }
                    }
                }
            });
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected float or Vector"))
    }

    pub fn matmul_array(&self, other: &Array) -> Array {
        let nd1 = self.ndim();
        let nd2 = other.ndim();

        // 2D Fast Path
        if nd1 == 2 && nd2 == 2 {
            let (r1, c1) = (self.nrows(), self.ncols());
            let (r2, c2) = (other.nrows(), other.ncols());
            if c1 != r2 { panic!("MatMul dim mismatch: {}x{} @ {}x{}", r1, c1, r2, c2); }
            let a = self.to_mat();
            let b = other.to_mat();
            return Self::from_mat(a * b);
        }

        // Batch MatMul (Rank 3+)
        let batch_shape1 = if nd1 > 2 { &self.shape[..nd1-2] } else { &[] };
        let batch_shape2 = if nd2 > 2 { &other.shape[..nd2-2] } else { &[] };
        
        let res_batch = Array::broadcast_shapes(batch_shape1, batch_shape2).unwrap();
        let m = self.shape[nd1-2];
        let k = self.shape[nd1-1];
        let k2 = other.shape[nd2-2];
        let n = other.shape[nd2-1];
        if k != k2 { panic!("MatMul inner dim mismatch: {} vs {}", k, k2); }

        let mut res_shape = res_batch.clone();
        res_shape.push(m);
        res_shape.push(n);
        
        let batch_count: usize = res_batch.iter().product();
        let slice_size = m * n;
        let mut out_data = vec![0.0; batch_count * slice_size];
        
        // Compute each slice
        for i in 0..batch_count {
            let slice1 = self.get_batch_slice(i, &res_batch, batch_shape1, m, k);
            let slice2 = other.get_batch_slice(i, &res_batch, batch_shape2, k, n);
            
            let m1 = slice1.to_mat();
            let m2 = slice2.to_mat();
            let res_mat = m1 * m2;
            
            let d = Array::from_mat(res_mat);
            out_data[i*slice_size..(i+1)*slice_size].copy_from_slice(d.data());
        }

        Self::from_flat(out_data, res_shape)
    }

    fn get_batch_slice(&self, batch_idx: usize, res_batch: &[usize], my_batch_shape: &[usize], r: usize, c: usize) -> Array {
        if my_batch_shape.is_empty() { return self.clone(); }
        
        let res_batch_strides = Array::compute_strides(res_batch);
        let my_batch_strides = Array::compute_strides(my_batch_shape);
        
        let mut my_idx = 0;
        let mut rem = batch_idx;
        let res_ndim = res_batch.len();
        let my_ndim = my_batch_shape.len();

        for d in 0..res_ndim {
            let coord = rem / res_batch_strides[d];
            rem %= res_batch_strides[d];
            
            if d >= res_ndim - my_ndim {
                let my_d = d - (res_ndim - my_ndim);
                if my_batch_shape[my_d] > 1 {
                    my_idx += coord * my_batch_strides[my_d];
                }
            }
        }
        
        let slice_len = r * c;
        let start = my_idx * slice_len;
        let data = self.data()[start..start+slice_len].to_vec();
        Array::from_flat(data, vec![r, c])
    }

    pub fn matmul_vec(&self, v: &Vector) -> Vec<f64> {
        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        v.with_slice(|s| {
            (0..r).into_par_iter().map(|i| {
                (0..c).map(|j| d[i*c+j] * s[j]).sum()
            }).collect()
        })
    }
}
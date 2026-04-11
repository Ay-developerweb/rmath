use pyo3::prelude::*;
use faer::{Mat};
use faer::linalg::solvers::DenseSolveCore;
use crate::vector::Vector;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Array {
    pub storage: ArrayStorage,
}

#[derive(Clone)]
pub enum ArrayStorage {
    Inline([f64; 16], usize, usize),
    Heap(Mat<f64>),
}

impl Array {
    pub fn nrows(&self) -> usize { match &self.storage { ArrayStorage::Inline(_, r, _) => *r, ArrayStorage::Heap(m) => m.nrows() } }
    pub fn ncols(&self) -> usize { match &self.storage { ArrayStorage::Inline(_, _, c) => *c, ArrayStorage::Heap(m) => m.ncols() } }

    #[inline(always)]
    pub fn with_mat_ref<R, F>(&self, f: F) -> R where F: FnOnce(faer::MatRef<'_, f64>) -> R {
        match &self.storage {
            ArrayStorage::Inline(data, r, c) => {
                let mut m = Mat::zeros(*r, *c);
                for i in 0..*r { for j in 0..*c { m[(i, j)] = data[i * c + j]; } }
                f(m.as_ref())
            }
            ArrayStorage::Heap(m) => f(m.as_ref()),
        }
    }

    pub fn to_mat(&self) -> Mat<f64> { self.with_mat_ref(|m| m.to_owned()) }
    pub fn from_mat(m: Mat<f64>) -> Self {
        let r = m.nrows();
        let c = m.ncols();
        if r * c <= 16 && r > 0 && c > 0 {
            let mut data = [0.0; 16];
            for i in 0..r { for j in 0..c { data[i * c + j] = m[(i, j)]; } }
            Array { storage: ArrayStorage::Inline(data, r, c) }
        } else { Array { storage: ArrayStorage::Heap(m) } }
    }
}

#[pymethods]
impl Array {
    #[new]
    pub fn new(data: Vec<Vec<f64>>) -> PyResult<Self> {
        let rows = data.len();
        if rows == 0 { return Ok(Array::from_mat(Mat::zeros(0, 0))); }
        let cols = data[0].len();
        let mut m = Mat::zeros(rows, cols);
        for i in 0..rows {
            if data[i].len() != cols { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
            for j in 0..cols { m[(i, j)] = data[i][j]; }
        }
        Ok(Array::from_mat(m))
    }

    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> Array { Array::from_mat(Mat::zeros(rows, cols)) }
    #[staticmethod]
    pub fn ones(rows: usize, cols: usize) -> Array {
        let mut m = Mat::zeros(rows, cols);
        for i in 0..rows { for j in 0..cols { m[(i, j)] = 1.0; } }
        Array::from_mat(m)
    }
    #[staticmethod]
    pub fn randn(rows: usize, cols: usize) -> Array {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        let mut rng = thread_rng();
        let mut m = Mat::zeros(rows, cols);
        for i in 0..rows { for j in 0..cols { m[(i, j)] = rng.sample(StandardNormal); } }
        Array::from_mat(m)
    }
    #[staticmethod]
    pub fn eye(n: usize) -> Array { Array::from_mat(Mat::identity(n, n)) }

    pub fn shape(&self) -> (usize, usize) { (self.nrows(), self.ncols()) }
    pub fn __repr__(&self) -> String { format!("Array(shape=({}, {}))", self.nrows(), self.ncols()) }

    pub fn to_list(&self) -> Vec<Vec<f64>> {
        self.with_mat_ref(|m| {
            let mut res = Vec::with_capacity(m.nrows());
            for i in 0..m.nrows() {
                let mut row = Vec::with_capacity(m.ncols());
                for j in 0..m.ncols() { row.push(m[(i, j)]); }
                res.push(row);
            }
            res
        })
    }
    pub fn tolist(&self) -> Vec<Vec<f64>> { self.to_list() }

    pub fn __add__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            if self.nrows() != other.nrows() || self.ncols() != other.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch")); }
            return self.with_mat_ref(|m1| { other.with_mat_ref(|m2| { Ok(Array::from_mat(m1 + m2)) })});
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if v.len_internal() != self.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Broadcasting len mismatch")); }
            let mut res = self.to_mat();
            v.with_slice(|s| { for j in 0..res.ncols() { let val = s[j]; for i in 0..res.nrows() { res[(i, j)] += val; } } });
            return Ok(Array::from_mat(res));
        }
        if let Ok(s) = rhs.extract::<f64>() {
            let mut res = self.to_mat();
            for j in 0..res.ncols() { for i in 0..res.nrows() { res[(i, j)] += s; } }
            return Ok(Array::from_mat(res));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid addition"))
    }

    pub fn __sub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            if self.nrows() != other.nrows() || self.ncols() != other.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch")); }
            return self.with_mat_ref(|m1| { other.with_mat_ref(|m2| { Ok(Array::from_mat(m1 - m2)) })});
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if v.len_internal() != self.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Broadcasting len mismatch")); }
            let mut res = self.to_mat();
            v.with_slice(|s| { for j in 0..res.ncols() { let val = s[j]; for i in 0..res.nrows() { res[(i, j)] -= val; } } });
            return Ok(Array::from_mat(res));
        }
        if let Ok(s) = rhs.extract::<f64>() {
            let mut res = self.to_mat();
            for j in 0..res.ncols() { for i in 0..res.nrows() { res[(i, j)] -= s; } }
            return Ok(Array::from_mat(res));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid subtraction"))
    }

    pub fn __mul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            let mut res = self.to_mat();
            for j in 0..res.ncols() { for i in 0..res.nrows() { res[(i, j)] *= s; } }
            return Ok(Array::from_mat(res));
        }
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            if self.nrows() != other.nrows() || self.ncols() != other.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch")); }
            return self.with_mat_ref(|m1| { other.with_mat_ref(|m2| {
                let mut res = Mat::zeros(m1.nrows(), m1.ncols());
                for j in 0..m1.ncols() { for i in 0..m1.nrows() { res[(i, j)] = m1[(i, j)] * m2[(i, j)]; } }
                Ok(Array::from_mat(res))
            })});
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid multiplication"))
    }

    pub fn __truediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("div by zero")); }
            let mut res = self.to_mat();
            for j in 0..res.ncols() { for i in 0..res.nrows() { res[(i, j)] /= s; } }
            return Ok(Array::from_mat(res));
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if v.len_internal() != self.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Broadcasting len mismatch")); }
            let mut res = self.to_mat();
            v.with_slice(|s| { for j in 0..res.ncols() { let val = s[j]; for i in 0..res.nrows() { res[(i, j)] /= val; } } });
            return Ok(Array::from_mat(res));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Invalid division"))
    }

    pub fn __isub__<'py>(&mut self, rhs: &Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if v.len_internal() != self.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
            match &mut self.storage {
                ArrayStorage::Inline(data, r, c) => { v.with_slice(|s| { for i in 0..*r { for j in 0..*c { data[i * *c + j] -= s[j]; } } }); }
                ArrayStorage::Heap(m) => { v.with_slice(|s| { for j in 0..m.ncols() { let val = s[j]; for i in 0..m.nrows() { m[(i, j)] -= val; } } }); }
            }
            return Ok(());
        }
        if let Ok(s) = rhs.extract::<f64>() {
            match &mut self.storage {
                ArrayStorage::Inline(data, r, c) => { for i in 0..*r { for j in 0..*c { data[i * *c + j] -= s; } } }
                ArrayStorage::Heap(m) => { for j in 0..m.ncols() { for i in 0..m.nrows() { m[(i, j)] -= s; } } }
            }
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Vector or float"))
    }

    pub fn __itruediv__<'py>(&mut self, rhs: &Bound<'py, PyAny>) -> PyResult<()> {
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if v.len_internal() != self.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Len mismatch")); }
            match &mut self.storage {
                ArrayStorage::Inline(data, r, c) => { v.with_slice(|s| { for i in 0..*r { for j in 0..*c { data[i * *c + j] /= s[j]; } } }); }
                ArrayStorage::Heap(m) => { v.with_slice(|s| { for j in 0..m.ncols() { let val = s[j]; for i in 0..m.nrows() { m[(i, j)] /= val; } } }); }
            }
            return Ok(());
        }
        if let Ok(s) = rhs.extract::<f64>() {
            if s == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("div by zero")); }
            match &mut self.storage {
                ArrayStorage::Inline(data, r, c) => { for i in 0..*r { for j in 0..*c { data[i * *c + j] /= s; } } }
                ArrayStorage::Heap(m) => { for j in 0..m.ncols() { for i in 0..m.nrows() { m[(i, j)] /= s; } } }
            }
            return Ok(());
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Vector or float"))
    }

    pub fn __matmul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        if let Ok(other) = rhs.extract::<PyRef<Array>>() {
            if self.ncols() != other.nrows() { return Err(pyo3::exceptions::PyValueError::new_err("Dim mismatch")); }
            return self.with_mat_ref(|m1| { other.with_mat_ref(|m2| {
                let res = Array::from_mat(m1 * m2);
                Ok(res.into_pyobject(py)?.into_any())
            })});
        }
        if let Ok(v) = rhs.extract::<PyRef<Vector>>() {
            if self.ncols() != v.len_internal() { return Err(pyo3::exceptions::PyValueError::new_err("Dim mismatch")); }
            return self.with_mat_ref(|m| {
                let v_res: Vec<f64> = v.with_slice(|s| {
                    (0..m.nrows()).into_par_iter().map(|i| {
                        let mut sum = 0.0;
                        for j in 0..m.ncols() { sum += m[(i, j)] * s[j]; }
                        sum
                    }).collect()
                });
                Ok(Vector::new(v_res).into_pyobject(py)?.into_any())
            });
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Array or Vector"))
    }

    pub fn matmul_trans(&self, v: &Vector) -> PyResult<Vector> {
        if self.nrows() != v.len_internal() { return Err(pyo3::exceptions::PyValueError::new_err("Dim mismatch for Transpose Matmul")); }
        self.with_mat_ref(|m| {
            let v_res: Vec<f64> = v.with_slice(|s| {
                (0..m.ncols()).into_par_iter().map(|j| { (0..m.nrows()).map(|i| m[(i, j)] * s[i]).sum() }).collect()
            });
            Ok(Vector::new(v_res))
        })
    }

    pub fn inv(&self) -> PyResult<Array> {
        self.with_mat_ref(|m| {
            if m.nrows() != m.ncols() { return Err(pyo3::exceptions::PyValueError::new_err("Must be square")); }
            Ok(Array::from_mat(m.partial_piv_lu().inverse()))
        })
    }

    pub fn transpose(&self) -> Self { self.with_mat_ref(|m| Array::from_mat(m.transpose().to_owned())) }
    pub fn abs(&self) -> Self { self.with_mat_ref(|m| { let mut res = m.to_owned(); for j in 0..res.ncols() { for i in 0..res.nrows() { res[(i, j)] = res[(i, j)].abs(); } } Array::from_mat(res) }) }
    pub fn mean(&self) -> f64 { self.with_mat_ref(|m| { let total: f64 = (0..m.ncols()).into_par_iter().map(|j| { (0..m.nrows()).map(|i| m[(i, j)]).sum::<f64>() }).sum(); total / (m.nrows() * m.ncols()) as f64 }) }

    #[pyo3(signature = (axis=None))]
    pub fn sum<'py>(&self, py: Python<'py>, axis: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.with_mat_ref(|m| {
            match axis {
                None => {
                    let total: f64 = (0..m.ncols()).into_par_iter().map(|j| { (0..m.nrows()).map(|i| m[(i, j)]).sum::<f64>() }).sum();
                    Ok(total.into_pyobject(py)?.into_any())
                }
                Some(0) => {
                    let col_sums: Vec<f64> = (0..m.ncols()).into_par_iter().map(|j| { (0..m.nrows()).map(|i| m[(i, j)]).sum() }).collect();
                    Ok(Vector::new(col_sums).into_pyobject(py)?.into_any())
                }
                Some(1) => {
                    let row_sums: Vec<f64> = (0..m.nrows()).into_par_iter().map(|i| { (0..m.ncols()).map(|j| m[(i, j)]).sum() }).collect();
                    Ok(Vector::new(row_sums).into_pyobject(py)?.into_any())
                }
                _ => Err(pyo3::exceptions::PyValueError::new_err("Axis invalid"))
            }
        })
    }

    pub fn mean_axis0(&self) -> PyResult<Vector> {
        let r = self.nrows();
        let c = self.ncols();
        if r == 0 { return Err(pyo3::exceptions::PyValueError::new_err("Empty")); }
        self.with_mat_ref(|m| {
            let sums: Vec<f64> = (0..c).into_par_iter().map(|j| { (0..r).map(|i| m[(i, j)]).sum::<f64>() / r as f64 }).collect();
            Ok(Vector::new(sums))
        })
    }

    pub fn std_axis0(&self) -> PyResult<Vector> {
        let r = self.nrows();
        let c = self.ncols();
        if r < 2 { return Err(pyo3::exceptions::PyValueError::new_err("r < 2")); }
        let mu = self.mean_axis0()?;
        self.with_mat_ref(|m| {
            let stds: Vec<f64> = mu.with_slice(|s| {
                (0..c).into_par_iter().map(|j| {
                    let mut sum_sq = 0.0;
                    for i in 0..r { sum_sq += (m[(i, j)] - s[j]).powi(2); }
                    (sum_sq / (r - 1) as f64).sqrt() + 1e-8
                }).collect()
            });
            Ok(Vector::new(stds))
        })
    }

    pub fn normalize(&self, mu: &Vector, sigma: &Vector) -> PyResult<Self> {
        let r = self.nrows();
        let c = self.ncols();
        if mu.len_internal() != c || sigma.len_internal() != c { return Err(pyo3::exceptions::PyValueError::new_err("Dim mismatch")); }
        self.with_mat_ref(|m| {
            let mut res = Mat::zeros(r, c);
            mu.with_slice(|mu_s| {
                sigma.with_slice(|sig_s| {
                    for j in 0..c {
                        let m_val = mu_s[j];
                        let s_val = sig_s[j];
                        for i in 0..r { res[(i, j)] = (m[(i, j)] - m_val) / s_val; }
                    }
                });
            });
            Ok(Array::from_mat(res))
        })
    }

    pub fn gram_matrix(&self) -> Self {
        self.with_mat_ref(|m| {
            // faer's MatRef * MatRef for Transpose is highly optimized and lazy!
            Array::from_mat(m.transpose() * m)
        })
    }

    pub fn covariance(&self, py: Python<'_>) -> PyResult<Self> {
        let r = self.nrows();
        if r < 2 { return Err(pyo3::exceptions::PyValueError::new_err("r < 2")); }
        let mu = self.mean_axis0()?;
        // Center the matrix (Self - Mu) using Python-token-safe conversion
        let mu_any = mu.into_pyobject(py)?.into_any();
        let centered = self.__sub__(&mu_any)?;
        // (X-mu).T * (X-mu) / (n-1)
        let gram = centered.gram_matrix();
        let scale = 1.0 / (r - 1) as f64;
        let scale_any = scale.into_pyobject(py)?.into_any();
        gram.__mul__(&scale_any)
    }

    pub fn sigmoid(&self) -> Self { self.with_mat_ref(|m| { let mut res = m.to_owned(); for j in 0..res.ncols() { for i in 0..res.nrows() { res[(i, j)] = 1.0 / (1.0 + (-res[(i, j)]).exp()); } } Array::from_mat(res) }) }
}

pub fn register_array(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Array>()?; Ok(()) }

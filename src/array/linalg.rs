use pyo3::prelude::*;
use faer::Mat;
use faer::linalg::solvers::{DenseSolveCore, Solve};
use super::core::Array;
use crate::vector::Vector;

/// Registers the Linear Algebra module (rmath.linalg).
///
/// This module provides high-performance matrix decompositions and solvers
/// backed by the `faer` and `matrixmultiply` Rust libraries.
///
/// Features:
/// - Matrix Inversion (LU Decomposition)
/// - SVD (Singular Value Decomposition)
/// - Eigenvalues/vectors (Symmetric Eigh)
/// - Batch Least Squares (Solve)
pub fn register_linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(rank, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(eigh, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_inv, m)?)?;
    Ok(())
}

#[pymethods]
impl Array {

    // ── Basic linear algebra ──────────────────────────────────────────────

    /// Compute the inverse of a square matrix.
    ///
    /// Uses LU decomposition with partial pivoting.
    ///
    /// Example:
    ///     >>> a = ra.Array([[1, 2], [3, 4]])
    ///     >>> a.inv()
    ///     Array([[-2.0, 1.0], [1.5, -0.5]])
    #[pyo3(name = "inv")]
    pub fn inv_py(&self, py: Python<'_>) -> PyResult<Self> {
        py.allow_threads(|| self.inv_internal())
    }

    /// Solve the linear system Ax = B for x.
    ///
    /// Example:
    ///     >>> a = ra.Array([[1, 2], [3, 4]])
    ///     >>> b = ra.Array([[5], [6]])
    ///     >>> x = a.solve(b)
    #[pyo3(name = "solve")]
    pub fn solve_py(&self, py: Python<'_>, b: &Array) -> PyResult<Self> {
        py.allow_threads(|| self.solve_internal(b))
    }

    /// Return the transpose of the matrix.
    #[pyo3(name = "transpose")]
    pub fn transpose_py(&self) -> Self {
        self.transpose_internal()
    }

    /// Shortcut for transpose.
    pub fn t(&self) -> Self { self.transpose_internal() }

    /// Compute the determinant of a square matrix.
    pub fn det(&self, py: Python<'_>) -> PyResult<f64> { py.allow_threads(|| self.det_internal()) }

    /// Estimate the rank of the matrix using SVD.
    pub fn rank(&self, py: Python<'_>) -> usize { py.allow_threads(|| self.rank_internal()) }

    /// Compute the trace (sum of diagonal elements) of a square matrix.
    pub fn trace(&self) -> PyResult<f64> {
        self.assert_square("trace")?;
        let n = self.nrows();
        let d = self.data();
        Ok((0..n).map(|i| d[i*n+i]).sum())
    }

    // ... other methods kept same for compatibility ...
    pub fn norm_frobenius(&self) -> f64 {
        self.data().iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute the QR decomposition of the matrix.
    ///
    /// Factors matrix A into an orthogonal matrix Q and an upper triangular matrix R.
    pub fn qr(&self, py: Python<'_>) -> PyResult<(Self, Self)> { py.allow_threads(|| self.qr_internal()) }

    /// Perform Singular Value Decomposition (SVD).
    ///
    /// Factors the matrix A into U, S, and V^T such that A = U * diag(S) * V^T.
    /// U and V^T are orthogonal matrices, and S contains the singular values.
    ///
    /// Returns:
    ///     A tuple (U, S, V_T) where U, V_T are Arrays and S is a Vector.
    pub fn svd(&self, py: Python<'_>) -> (Self, Vector, Self) { py.allow_threads(|| self.svd_internal()) }

    /// Compute the eigenvalues and eigenvectors of a symmetric matrix.
    ///
    /// Returns:
    ///     A tuple (vals, vecs) where vals is a Vector of eigenvalues
    ///     and vecs is an Array of eigenvectors (as columns).
    pub fn eigh(&self, py: Python<'_>) -> PyResult<(Vector, Self)> { py.allow_threads(|| self.eigh_internal()) }

    /// Compute the Cholesky decomposition of a positive-definite symmetric matrix.
    ///
    /// Returns the lower triangular matrix L such that A = LL^T.
    pub fn cholesky(&self, py: Python<'_>) -> PyResult<Self> { py.allow_threads(|| self.cholesky_internal()) }

    /// Compute the Moore-Penrose pseudo-inverse of the matrix using SVD.
    pub fn pseudo_inv(&self, py: Python<'_>) -> PyResult<Self> { py.allow_threads(|| self.pseudo_inv_internal()) }

    pub fn gram_matrix(&self) -> Self {
        let r = self.nrows();
        let c = self.ncols();
        let mut out = vec![0.0; c * c];
        let d = self.storage_slice();
        let offset = self.offset;
        let rs = self.strides[0];
        let cs = self.strides[1];
        
        unsafe {
            matrixmultiply::dgemm(
                c, r, c,
                1.0,
                d[offset..].as_ptr(), cs, rs, // Transposed strides
                d[offset..].as_ptr(), rs, cs, // Original strides
                0.0,
                out.as_mut_ptr(), c as isize, 1,
            );
        }
        Self::from_flat(out, vec![c, c])
    }

    pub fn covariance(&self, py: Python<'_>) -> PyResult<Self> {
        let r = self.nrows();
        if r < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Need at least 2 rows")); }
        let mu = self.mean_axis0()?;
        let centered = self.sub_array(&mu.into_array())?;
        let gram = centered.gram_matrix();
        let scale = 1.0 / (r - 1) as f64;
        let scale_any = scale.into_pyobject(py)?.into_any();
        gram.__mul__(&scale_any)
    }
}

// ── Functional API ───────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "inv")]
pub fn inv(py: Python<'_>, a: &Array) -> PyResult<Array> { py.allow_threads(|| a.inv_internal()) }

#[pyfunction]
#[pyo3(name = "det")]
pub fn det(py: Python<'_>, a: &Array) -> PyResult<f64> { py.allow_threads(|| a.det_internal()) }

#[pyfunction]
#[pyo3(name = "solve")]
pub fn solve(py: Python<'_>, a: &Array, b: &Array) -> PyResult<Array> { py.allow_threads(|| a.solve_internal(b)) }

#[pyfunction]
#[pyo3(name = "rank")]
pub fn rank(py: Python<'_>, a: &Array) -> usize { py.allow_threads(|| a.rank_internal()) }

#[pyfunction]
#[pyo3(name = "transpose")]
pub fn transpose(a: &Array) -> Array { a.transpose_internal() }

#[pyfunction]
#[pyo3(name = "qr")]
pub fn qr(py: Python<'_>, a: &Array) -> PyResult<(Array, Array)> { py.allow_threads(|| a.qr_internal()) }

#[pyfunction]
#[pyo3(name = "svd")]
pub fn svd(py: Python<'_>, a: &Array) -> (Array, Vector, Array) { py.allow_threads(|| a.svd_internal()) }

#[pyfunction]
#[pyo3(name = "eigh")]
pub fn eigh(py: Python<'_>, a: &Array) -> PyResult<(Vector, Array)> { py.allow_threads(|| a.eigh_internal()) }

#[pyfunction]
#[pyo3(name = "cholesky")]
pub fn cholesky(py: Python<'_>, a: &Array) -> PyResult<Array> { py.allow_threads(|| a.cholesky_internal()) }

#[pyfunction]
#[pyo3(name = "pseudo_inv")]
pub fn pseudo_inv(py: Python<'_>, a: &Array) -> PyResult<Array> { py.allow_threads(|| a.pseudo_inv_internal()) }

impl Array {
    pub fn inv_internal(&self) -> PyResult<Self> {
        self.assert_square("inv")?;
        let m = self.to_mat();
        Ok(Self::from_mat(m.partial_piv_lu().inverse()))
    }

    pub fn solve_internal(&self, b: &Array) -> PyResult<Self> {
        self.assert_square("solve")?;
        if b.nrows() != self.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err("b rows != A rows"));
        }
        let a = self.to_mat();
        let b_mat = b.to_mat();
        let x = a.partial_piv_lu().solve(&b_mat);
        Ok(Self::from_mat(x))
    }

    pub fn det_internal(&self) -> PyResult<f64> {
        self.assert_square("det")?;
        let m = self.to_mat();
        Ok(m.determinant())
    }

    pub fn rank_internal(&self) -> usize {
        let m = self.to_mat();
        let svd = m.thin_svd().unwrap();
        let s = svd.S().column_vector();
        let tol = 1e-10 * s[0].abs().max(1.0);
        (0..s.nrows()).filter(|&i| s[i].abs() > tol).count()
    }

    pub fn qr_internal(&self) -> PyResult<(Self, Self)> {
        let m = self.to_mat();
        let qr = m.col_piv_qr();
        let q_mat = qr.compute_thin_Q();
        let r_mat = q_mat.as_ref().transpose() * m.as_ref();
        Ok((Self::from_mat(q_mat), Self::from_mat(r_mat)))
    }

    pub fn svd_internal(&self) -> (Self, Vector, Self) {
        let m = self.to_mat();
        let svd = m.thin_svd().unwrap();
        let u = Self::from_mat(svd.U().to_owned());
        let s_col = svd.S().column_vector();
        let k = s_col.nrows();
        let s_vec: Vec<f64> = (0..k).map(|i| s_col[i]).collect();
        let vt = Self::from_mat(svd.V().to_owned().transpose().to_owned());
        (u, Vector::new(s_vec), vt)
    }

    pub fn eigh_internal(&self) -> PyResult<(Vector, Self)> {
        self.assert_square("eigh")?;
        let m = self.to_mat();
        let eig = m.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let n = self.nrows();
        let vals: Vec<f64> = (0..n).map(|i| eig.S().column_vector()[i]).collect();
        let vecs = Self::from_mat(eig.U().to_owned());
        Ok((Vector::new(vals), vecs))
    }

    pub fn cholesky_internal(&self) -> PyResult<Self> {
        self.assert_square("cholesky")?;
        let m = self.to_mat();
        match m.llt(faer::Side::Lower) {
            Ok(_) => {
                let n = self.nrows();
                let mut l = Mat::<f64>::zeros(n, n);
                let d = self.data();
                for i in 0..n {
                    for j in 0..=i {
                        let mut sum = d[i * n + j];
                        for k in 0..j { sum -= l[(i, k)] * l[(j, k)]; }
                        if i == j {
                            if sum <= 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("Not PD")); }
                            l[(i, j)] = sum.sqrt();
                        } else { l[(i, j)] = sum / l[(j, j)]; }
                    }
                }
                Ok(Self::from_mat(l))
            }
            Err(_) => Err(pyo3::exceptions::PyValueError::new_err("Not PD"))
        }
    }

    pub fn pseudo_inv_internal(&self) -> PyResult<Self> {
        let (u, s_vec, vt) = self.svd_internal();
        let tol = 1e-10;
        let k = s_vec.len_internal();
        let mut s_inv_data = vec![0.0; k * k];
        s_vec.with_slice(|s| {
            for i in 0..k {
                s_inv_data[i*k+i] = if s[i].abs() > tol { 1.0 / s[i] } else { 0.0 };
            }
        });
        let s_inv = Self::from_flat(s_inv_data, vec![k, k]);
        let ut = u.transpose_internal();
        let vs = vt.transpose_internal().matmul_array(&s_inv);
        Ok(vs.matmul_array(&ut))
    }

    fn assert_square(&self, op: &str) -> PyResult<()> {
        if self.nrows() != self.ncols() {
            Err(pyo3::exceptions::PyValueError::new_err(
                format!("{} requires square matrix, got {:?}", op, self.shape)))
        } else { Ok(()) }
    }
}
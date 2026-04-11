use pyo3::prelude::*;
use rayon::prelude::*;

// ============================================================================
// --- Vector Class ---
// ============================================================================

#[pyclass]
#[derive(Clone)]
pub struct Vector {
    pub data: Vec<f64>,
}

#[pymethods]
impl Vector {
    #[new]
    pub fn new(data: Vec<f64>) -> Self {
        Vector { data }
    }

    // -----------------------------------------------------------------------
    // --- Python protocol methods ---
    // -----------------------------------------------------------------------

    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    pub fn __repr__(&self) -> String {
        if self.data.len() <= 6 {
            format!("Vector({:?})", self.data)
        } else {
            format!(
                "Vector([{}, {}, {}, ..., {}, {}, {}], len={})",
                self.data[0],
                self.data[1],
                self.data[2],
                self.data[self.data.len() - 3],
                self.data[self.data.len() - 2],
                self.data[self.data.len() - 1],
                self.data.len()
            )
        }
    }

    // -----------------------------------------------------------------------
    // --- Dunder arithmetic operators (scalar float OR Vector dispatch) ---
    // -----------------------------------------------------------------------

    /// v + scalar  OR  v + v2  (element-wise)
    pub fn __add__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            return Ok(self.add_scalar(s));
        }
        if let Ok(other) = rhs.extract::<PyRef<Vector>>() {
            return self.add_vec(&other);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Vector.__add__: expected float or Vector",
        ))
    }

    /// scalar + v  (reflected)
    pub fn __radd__(&self, lhs: f64) -> Self {
        self.add_scalar(lhs)
    }

    /// v - scalar  OR  v - v2  (element-wise)
    pub fn __sub__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            return Ok(self.sub_scalar(s));
        }
        if let Ok(other) = rhs.extract::<PyRef<Vector>>() {
            return self.sub_vec(&other);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Vector.__sub__: expected float or Vector",
        ))
    }

    /// scalar - v  (reflected)
    pub fn __rsub__(&self, lhs: f64) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| lhs - x).collect(),
        }
    }

    /// v * scalar  OR  v * v2  (element-wise)
    pub fn __mul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            return Ok(self.mul_scalar(s));
        }
        if let Ok(other) = rhs.extract::<PyRef<Vector>>() {
            return self.mul_vec(&other);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Vector.__mul__: expected float or Vector",
        ))
    }

    /// scalar * v  (reflected)
    pub fn __rmul__(&self, lhs: f64) -> Self {
        self.mul_scalar(lhs)
    }

    /// v / scalar  OR  v / v2  (element-wise)
    pub fn __truediv__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = rhs.extract::<f64>() {
            return self.div_scalar(s);
        }
        if let Ok(other) = rhs.extract::<PyRef<Vector>>() {
            return self.div_vec(&other);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Vector.__truediv__: expected float or Vector",
        ))
    }

    /// -v  (element-wise negation)
    pub fn __neg__(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| -x).collect(),
        }
    }

    /// v @ v2  — dot product (scalar result) via the @ operator
    pub fn __matmul__(&self, other: &Vector) -> PyResult<f64> {
        self.dot(other)
    }

    // -----------------------------------------------------------------------
    // --- Constructors ---
    // -----------------------------------------------------------------------

    /// Creates a Vector with a range of numbers [start, stop) directly in Rust memory.
    /// Supports: Vector.range(stop) or Vector.range(start, stop, step)
    #[pyo3(signature = (start_or_stop, stop=None, step=1.0))]
    #[staticmethod]
    pub fn range(start_or_stop: f64, stop: Option<f64>, step: f64) -> PyResult<Self> {
        let (start, stop) = match stop {
            Some(s) => (start_or_stop, s),
            None => (0.0, start_or_stop),
        };
        if step == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "step must not be zero",
            ));
        }
        let n_f64 = ((stop - start) / step).ceil().max(0.0);
        if n_f64 > usize::MAX as f64 {
            return Err(pyo3::exceptions::PyOverflowError::new_err(
                "Range exceeds maximum pointer-sized integer",
            ));
        }
        let n = n_f64 as usize;
        let mut data: Vec<f64> = Vec::new();
        data.try_reserve(n).map_err(|_| {
            pyo3::exceptions::PyMemoryError::new_err("Failed to allocate Vector memory")
        })?;
        let data: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| start + i as f64 * step)
            .collect();
        Ok(Vector { data })
    }

    /// Creates a Vector with a specified number of linearly spaced points [start, stop].
    #[staticmethod]
    pub fn linspace(start: f64, stop: f64, num: usize) -> PyResult<Self> {
        if num == 0 {
            return Ok(Vector { data: Vec::new() });
        }
        if num == 1 {
            return Ok(Vector { data: vec![start] });
        }
        let mut data: Vec<f64> = Vec::new();
        data.try_reserve(num).map_err(|_| {
            pyo3::exceptions::PyMemoryError::new_err("Failed to allocate Vector memory")
        })?;
        let step = (stop - start) / (num - 1) as f64;
        let data: Vec<f64> = (0..num)
            .into_par_iter()
            .map(|i| start + i as f64 * step)
            .collect();
        Ok(Vector { data })
    }

    /// Instantly sums a range [start, stop) using the arithmetic progression formula.
    /// Supports: Vector.sum_range(stop) or Vector.sum_range(start, stop, step)
    #[pyo3(signature = (start_or_stop, stop=None, step=1.0))]
    #[staticmethod]
    pub fn sum_range(start_or_stop: f64, stop: Option<f64>, step: f64) -> f64 {
        let (start, stop) = match stop {
            Some(s) => (start_or_stop, s),
            None => (0.0, start_or_stop),
        };
        if step == 0.0 || (stop > start && step < 0.0) || (stop < start && step > 0.0) {
            return 0.0;
        }
        let n = ((stop - start) / step).ceil().max(0.0);
        if n <= 0.0 {
            return 0.0;
        }
        let last = start + (n - 1.0) * step;
        (n / 2.0) * (start + last)
    }

    // -----------------------------------------------------------------------
    // --- Conversion ---
    // -----------------------------------------------------------------------

    pub fn to_list(&self) -> Vec<f64> {
        self.data.clone()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // -----------------------------------------------------------------------
    // --- Scalar arithmetic (element-wise, scalar arg) ---
    // -----------------------------------------------------------------------

    pub fn add_scalar(&self, s: f64) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x + s).collect(),
        }
    }

    pub fn sub_scalar(&self, s: f64) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x - s).collect(),
        }
    }

    pub fn mul_scalar(&self, s: f64) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x * s).collect(),
        }
    }

    pub fn div_scalar(&self, s: f64) -> PyResult<Self> {
        if s == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "division by zero",
            ));
        }
        Ok(Vector {
            data: self.data.par_iter().map(|&x| x / s).collect(),
        })
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.powf(exp)).collect(),
        }
    }

    /// Clamps every element to [lo, hi].
    pub fn clamp(&self, lo: f64, hi: f64) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.clamp(lo, hi)).collect(),
        }
    }

    // -----------------------------------------------------------------------
    // --- Vector-Vector element-wise ops ---
    // -----------------------------------------------------------------------

    /// Element-wise addition of two Vectors of equal length.
    pub fn add_vec(&self, other: &Vector) -> PyResult<Self> {
        if self.data.len() != other.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vectors must have the same length for element-wise addition",
            ));
        }
        Ok(Vector {
            data: self
                .data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        })
    }

    /// Element-wise subtraction of two Vectors of equal length.
    pub fn sub_vec(&self, other: &Vector) -> PyResult<Self> {
        if self.data.len() != other.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vectors must have the same length for element-wise subtraction",
            ));
        }
        Ok(Vector {
            data: self
                .data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        })
    }

    /// Element-wise multiplication of two Vectors of equal length.
    pub fn mul_vec(&self, other: &Vector) -> PyResult<Self> {
        if self.data.len() != other.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vectors must have the same length for element-wise multiplication",
            ));
        }
        Ok(Vector {
            data: self
                .data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a * b)
                .collect(),
        })
    }

    /// Element-wise division of two Vectors of equal length.
    pub fn div_vec(&self, other: &Vector) -> PyResult<Self> {
        if self.data.len() != other.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vectors must have the same length for element-wise division",
            ));
        }
        // Check for zero divisors first
        if other.data.iter().any(|&x| x == 0.0) {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "Element-wise division: divisor vector contains zero",
            ));
        }
        Ok(Vector {
            data: self
                .data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a / b)
                .collect(),
        })
    }

    /// Dot product of two equal-length vectors.
    pub fn dot(&self, other: &Vector) -> PyResult<f64> {
        if self.data.len() != other.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vectors must have the same length for dot product",
            ));
        }
        Ok(self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(&a, &b)| a * b)
            .sum())
    }

    // -----------------------------------------------------------------------
    // --- Rounding & Absolute value ---
    // -----------------------------------------------------------------------

    pub fn ceil(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.ceil()).collect(),
        }
    }
    pub fn floor(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.floor()).collect(),
        }
    }
    pub fn round(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.round()).collect(),
        }
    }
    pub fn trunc(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.trunc()).collect(),
        }
    }
    pub fn abs(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.abs()).collect(),
        }
    }
    pub fn sqrt(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.sqrt()).collect(),
        }
    }
    pub fn cbrt(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.cbrt()).collect(),
        }
    }

    // -----------------------------------------------------------------------
    // --- Trigonometry ---
    // -----------------------------------------------------------------------

    pub fn sin(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.sin()).collect(),
        }
    }
    pub fn cos(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.cos()).collect(),
        }
    }
    pub fn tan(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.tan()).collect(),
        }
    }
    pub fn asin(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.asin()).collect(),
        }
    }
    pub fn acos(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.acos()).collect(),
        }
    }
    pub fn atan(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.atan()).collect(),
        }
    }

    // -----------------------------------------------------------------------
    // --- Hyperbolic ---
    // -----------------------------------------------------------------------

    pub fn sinh(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.sinh()).collect(),
        }
    }
    pub fn cosh(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.cosh()).collect(),
        }
    }
    pub fn tanh(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.tanh()).collect(),
        }
    }

    // -----------------------------------------------------------------------
    // --- Exponential & Logarithmic ---
    // -----------------------------------------------------------------------

    pub fn exp(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.exp()).collect(),
        }
    }
    pub fn exp2(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.exp2()).collect(),
        }
    }
    pub fn expm1(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.exp_m1()).collect(),
        }
    }
    pub fn log(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.ln()).collect(),
        }
    }
    pub fn log2(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.log2()).collect(),
        }
    }
    pub fn log10(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.log10()).collect(),
        }
    }
    pub fn log1p(&self) -> Self {
        Vector {
            data: self.data.par_iter().map(|&x| x.ln_1p()).collect(),
        }
    }

    // -----------------------------------------------------------------------
    // --- Statistical reductions (scalar results) ---
    // -----------------------------------------------------------------------

    pub fn sum(&self) -> f64 {
        self.data.par_iter().sum()
    }

    pub fn prod(&self) -> f64 {
        self.data.par_iter().product()
    }

    pub fn mean(&self) -> PyResult<f64> {
        if self.data.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot calculate mean of an empty Vector",
            ));
        }
        Ok(self.data.par_iter().sum::<f64>() / self.data.len() as f64)
    }

    pub fn variance(&self) -> PyResult<f64> {
        let n = self.data.len();
        if n < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Variance requires at least 2 elements",
            ));
        }
        let m = self.mean()?;
        let sum_sq: f64 = self.data.par_iter().map(|&x| (x - m).powi(2)).sum();
        Ok(sum_sq / (n - 1) as f64)
    }

    pub fn std_dev(&self) -> PyResult<f64> {
        Ok(self.variance()?.sqrt())
    }

    pub fn min(&self) -> PyResult<f64> {
        if self.data.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot find minimum of an empty Vector",
            ));
        }
        Ok(self
            .data
            .par_iter()
            .cloned()
            .reduce(|| f64::INFINITY, f64::min))
    }

    pub fn max(&self) -> PyResult<f64> {
        if self.data.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot find maximum of an empty Vector",
            ));
        }
        Ok(self
            .data
            .par_iter()
            .cloned()
            .reduce(|| f64::NEG_INFINITY, f64::max))
    }

    // -----------------------------------------------------------------------
    // --- Boolean predicates ---
    // -----------------------------------------------------------------------

    pub fn isnan(&self) -> Vec<bool> {
        self.data.par_iter().map(|&x| x.is_nan()).collect()
    }
    pub fn isfinite(&self) -> Vec<bool> {
        self.data.par_iter().map(|&x| x.is_finite()).collect()
    }
    pub fn isinf(&self) -> Vec<bool> {
        self.data.par_iter().map(|&x| x.is_infinite()).collect()
    }
    pub fn is_integer(&self) -> Vec<bool> {
        self.data.par_iter().map(|&x| x.fract() == 0.0).collect()
    }
    pub fn is_prime(&self) -> Vec<bool> {
        self.data
            .par_iter()
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
                for i in (3..=limit).step_by(2) {
                    if n % i == 0 {
                        return false;
                    }
                }
                true
            })
            .collect()
    }
}

// ============================================================================
// --- Functional API (rmath.vector.*) ---
// ============================================================================

#[pyfunction]
#[pyo3(signature = (start_or_stop, stop=None, step=1.0))]
pub fn range(start_or_stop: f64, stop: Option<f64>, step: f64) -> PyResult<Vector> {
    Vector::range(start_or_stop, stop, step)
}

#[pyfunction]
pub fn linspace(start: f64, stop: f64, num: usize) -> PyResult<Vector> {
    Vector::linspace(start, stop, num)
}

#[pyfunction]
#[pyo3(signature = (start_or_stop, stop=None, step=1.0))]
pub fn sum_range(start_or_stop: f64, stop: Option<f64>, step: f64) -> f64 {
    Vector::sum_range(start_or_stop, stop, step)
}

#[pyfunction]
pub fn add_scalar(a: Vec<f64>, s: f64) -> Vec<f64> {
    Vector::new(a).add_scalar(s).to_list()
}
#[pyfunction]
pub fn sub_scalar(a: Vec<f64>, s: f64) -> Vec<f64> {
    Vector::new(a).sub_scalar(s).to_list()
}
#[pyfunction]
pub fn mul_scalar(a: Vec<f64>, s: f64) -> Vec<f64> {
    Vector::new(a).mul_scalar(s).to_list()
}
#[pyfunction]
pub fn div_scalar(a: Vec<f64>, s: f64) -> PyResult<Vec<f64>> {
    Ok(Vector::new(a).div_scalar(s)?.to_list())
}

#[pyfunction]
pub fn add_vec(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(Vector::new(a).add_vec(&Vector::new(b))?.to_list())
}
#[pyfunction]
pub fn sub_vec(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(Vector::new(a).sub_vec(&Vector::new(b))?.to_list())
}
#[pyfunction]
pub fn mul_vec(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(Vector::new(a).mul_vec(&Vector::new(b))?.to_list())
}
#[pyfunction]
pub fn div_vec(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(Vector::new(a).div_vec(&Vector::new(b))?.to_list())
}
#[pyfunction]
pub fn dot(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    Vector::new(a).dot(&Vector::new(b))
}

#[pyfunction]
pub fn ceil(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).ceil().to_list()
}
#[pyfunction]
pub fn floor(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).floor().to_list()
}
#[pyfunction]
pub fn round(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).round().to_list()
}
#[pyfunction]
pub fn trunc(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).trunc().to_list()
}
#[pyfunction]
pub fn abs(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).abs().to_list()
}
#[pyfunction]
pub fn sqrt(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).sqrt().to_list()
}
#[pyfunction]
pub fn cbrt(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).cbrt().to_list()
}

#[pyfunction]
pub fn sin(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).sin().to_list()
}
#[pyfunction]
pub fn cos(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).cos().to_list()
}
#[pyfunction]
pub fn tan(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).tan().to_list()
}
#[pyfunction]
pub fn asin(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).asin().to_list()
}
#[pyfunction]
pub fn acos(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).acos().to_list()
}
#[pyfunction]
pub fn atan(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).atan().to_list()
}
#[pyfunction]
pub fn sinh(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).sinh().to_list()
}
#[pyfunction]
pub fn cosh(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).cosh().to_list()
}
#[pyfunction]
pub fn tanh(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).tanh().to_list()
}

#[pyfunction]
pub fn exp(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).exp().to_list()
}
#[pyfunction]
pub fn exp2(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).exp2().to_list()
}
#[pyfunction]
pub fn expm1(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).expm1().to_list()
}
#[pyfunction]
pub fn log(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).log().to_list()
}
#[pyfunction]
pub fn log2(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).log2().to_list()
}
#[pyfunction]
pub fn log10(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).log10().to_list()
}
#[pyfunction]
pub fn log1p(a: Vec<f64>) -> Vec<f64> {
    Vector::new(a).log1p().to_list()
}

#[pyfunction]
pub fn sum(a: Vec<f64>) -> f64 {
    Vector::new(a).sum()
}
#[pyfunction]
pub fn prod(a: Vec<f64>) -> f64 {
    Vector::new(a).prod()
}
#[pyfunction]
pub fn mean(a: Vec<f64>) -> PyResult<f64> {
    Vector::new(a).mean()
}
#[pyfunction]
pub fn variance(a: Vec<f64>) -> PyResult<f64> {
    Vector::new(a).variance()
}
#[pyfunction]
pub fn std_dev(a: Vec<f64>) -> PyResult<f64> {
    Vector::new(a).std_dev()
}
#[pyfunction]
pub fn min(a: Vec<f64>) -> PyResult<f64> {
    Vector::new(a).min()
}
#[pyfunction]
pub fn max(a: Vec<f64>) -> PyResult<f64> {
    Vector::new(a).max()
}

#[pyfunction]
pub fn isnan(a: Vec<f64>) -> Vec<bool> {
    Vector::new(a).isnan()
}
#[pyfunction]
pub fn isfinite(a: Vec<f64>) -> Vec<bool> {
    Vector::new(a).isfinite()
}
#[pyfunction]
pub fn isinf(a: Vec<f64>) -> Vec<bool> {
    Vector::new(a).isinf()
}
#[pyfunction]
pub fn is_integer(a: Vec<f64>) -> Vec<bool> {
    Vector::new(a).is_integer()
}
#[pyfunction]
pub fn is_prime(a: Vec<f64>) -> Vec<bool> {
    Vector::new(a).is_prime()
}

#[pyfunction]
pub fn clamp(a: Vec<f64>, min: f64, max: f64) -> Vec<f64> {
    a.par_iter().map(|&x| x.clamp(min, max)).collect()
}

// ============================================================================
// --- Registration ---
// ============================================================================

pub fn register_vector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Class
    m.add_class::<Vector>()?;

    // Constructors / generators
    m.add_function(wrap_pyfunction!(range, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(sum_range, m)?)?;

    // Scalar arithmetic
    m.add_function(wrap_pyfunction!(add_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(sub_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(mul_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(div_scalar, m)?)?;

    // Vector-vector ops
    m.add_function(wrap_pyfunction!(add_vec, m)?)?;
    m.add_function(wrap_pyfunction!(sub_vec, m)?)?;
    m.add_function(wrap_pyfunction!(mul_vec, m)?)?;
    m.add_function(wrap_pyfunction!(div_vec, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;

    // Rounding / Transforms
    m.add_function(wrap_pyfunction!(ceil, m)?)?;
    m.add_function(wrap_pyfunction!(floor, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)?;
    m.add_function(wrap_pyfunction!(trunc, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(cbrt, m)?)?;

    // Trigonometry
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(asin, m)?)?;
    m.add_function(wrap_pyfunction!(acos, m)?)?;
    m.add_function(wrap_pyfunction!(atan, m)?)?;

    // Hyperbolic
    m.add_function(wrap_pyfunction!(sinh, m)?)?;
    m.add_function(wrap_pyfunction!(cosh, m)?)?;
    m.add_function(wrap_pyfunction!(tanh, m)?)?;

    // Exponential / Log
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(exp2, m)?)?;
    m.add_function(wrap_pyfunction!(expm1, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(log2, m)?)?;
    m.add_function(wrap_pyfunction!(log10, m)?)?;
    m.add_function(wrap_pyfunction!(log1p, m)?)?;

    // Reductions
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;

    // Predicates
    m.add_function(wrap_pyfunction!(isnan, m)?)?;
    m.add_function(wrap_pyfunction!(isfinite, m)?)?;
    m.add_function(wrap_pyfunction!(isinf, m)?)?;
    m.add_function(wrap_pyfunction!(is_integer, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;

    Ok(())
}

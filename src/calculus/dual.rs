use pyo3::prelude::*;

/// Dual numbers: a + b*epsilon where epsilon^2 = 0.
///
/// Dual numbers are the foundation of forward-mode Automatic Differentiation (AD).
/// When a function is evaluated with a `Dual(x, 1.0)`, the resulting `Dual`
/// value contains `f(x)` in its value and `f'(x)` in its derivative.
///
/// Examples:
///     >>> from rmath.calculus import Dual
///     >>> d = Dual(3.0, 1.0)
///     >>> f = lambda x: x**2 + 5*x
///     >>> res = f(d)
///     >>> res.value        # 3^2 + 5*3 = 9 + 15 = 24
///     24.0
///     >>> res.derivative   # 2*3 + 5 = 11
///     11.0
#[pyclass(module = "rmath")]
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub val: f64,
    pub der: f64,
}

#[pymethods]
impl Dual {
    /// Create a new Dual number: a + b*ε.
    #[new]
    pub fn new(val: f64, der: f64) -> Self {
        Dual { val, der }
    }

    #[getter]
    pub fn value(&self) -> f64 { self.val }
    #[getter]
    pub fn derivative(&self) -> f64 { self.der }

    pub fn __repr__(&self) -> String {
        format!("Dual(val={}, der={})", self.val, self.der)
    }

    // --- Basic Arithmetic (Overloaded in Python) ---
    pub fn __add__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        let o = extract_dual(other)?;
        Ok(Dual { val: self.val + o.val, der: self.der + o.der })
    }

    pub fn __sub__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        let o = extract_dual(other)?;
        Ok(Dual { val: self.val - o.val, der: self.der - o.der })
    }

    pub fn __mul__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        let o = extract_dual(other)?;
        // (a + bε)(c + dε) = ac + (ad + bc)ε
        Ok(Dual { val: self.val * o.val, der: self.val * o.der + self.der * o.val })
    }

    pub fn __truediv__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        let o = extract_dual(other)?;
        if o.val == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("div by zero")); }
        // (a+bε)/(c+dε) = (a/c) + ((bc - ad)/c^2)ε
        let val = self.val / o.val;
        let der = (self.der * o.val - self.val * o.der) / (o.val * o.val);
        Ok(Dual { val, der })
    }

    pub fn __rtruediv__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        let o = extract_dual(other)?;
        if self.val == 0.0 { return Err(pyo3::exceptions::PyZeroDivisionError::new_err("div by zero")); }
        // (c+dε)/(a+bε) = (c/a) + ((da - cb)/a^2)ε
        let val = o.val / self.val;
        let der = (o.der * self.val - o.val * self.der) / (self.val * self.val);
        Ok(Dual { val, der })
    }

    pub fn __radd__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        self.__add__(other)
    }

    pub fn __rsub__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        let o = extract_dual(other)?;
        Ok(Dual { val: o.val - self.val, der: o.der - self.der })
    }

    pub fn __rmul__(&self, other: Bound<'_, PyAny>) -> PyResult<Dual> {
        self.__mul__(other)
    }


    pub fn __pow__(&self, p: f64, _mod: Option<f64>) -> Dual {
        // d/dx(u^p) = p * u^(p-1) * du/dx
        Dual { 
            val: self.val.powf(p), 
            der: p * self.val.powf(p - 1.0) * self.der 
        }
    }

    pub fn __neg__(&self) -> Dual {
        Dual { val: -self.val, der: -self.der }
    }

    // --- Transcendental Operations ---

    /// Sine of a dual number.
    /// Derivative: cos(u) * du/dx
    ///
    /// Examples:
    ///     >>> from rmath.calculus import Dual
    ///     >>> Dual(0.0, 1.0).sin()
    ///     Dual(val=0.0, der=1.0)
    pub fn sin(&self) -> Dual {
        Dual { val: self.val.sin(), der: self.val.cos() * self.der }
    }

    /// Cosine of a dual number.
    /// Derivative: -sin(u) * du/dx
    ///
    /// Examples:
    ///     >>> from rmath.calculus import Dual
    ///     >>> Dual(0.0, 1.0).cos()
    ///     Dual(val=1.0, der=-0.0)
    pub fn cos(&self) -> Dual {
        Dual { val: self.val.cos(), der: -self.val.sin() * self.der }
    }

    /// Natural exponential of a dual number.
    /// Derivative: exp(u) * du/dx
    ///
    /// Examples:
    ///     >>> from rmath.calculus import Dual
    ///     >>> Dual(1.0, 1.0).exp()
    ///     Dual(val=2.71828..., der=2.71828...)
    pub fn exp(&self) -> Dual {
        let ev = self.val.exp();
        Dual { val: ev, der: ev * self.der }
    }

    /// Natural logarithm of a dual number.
    /// Derivative: (1/u) * du/dx
    ///
    /// Examples:
    ///     >>> from rmath.calculus import Dual
    ///     >>> Dual(1.0, 1.0).log()
    ///     Dual(val=0.0, der=1.0)
    pub fn log(&self) -> Dual {
        Dual { val: self.val.ln(), der: self.der / self.val }
    }

    /// Derivative: 2/sqrt(pi) * exp(-u^2) * du/dx
    pub fn erf(&self) -> Dual {
        let val = statrs::function::erf::erf(self.val);
        let pi = std::f64::consts::PI;
        let der = (2.0 / pi.sqrt()) * (-self.val * self.val).exp() * self.der;
        Dual { val, der }
    }

    /// Derivative: gamma(u) * digamma(u) * du/dx
    pub fn gamma(&self) -> Dual {
        let val = statrs::function::gamma::gamma(self.val);
        let der = val * statrs::function::gamma::digamma(self.val) * self.der;
        Dual { val, der }
    }
}

// --- Internal Helper ---
fn extract_dual(obj: Bound<'_, PyAny>) -> PyResult<Dual> {
    if let Ok(d) = obj.extract::<Dual>() {
        Ok(d)
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(Dual { val: f, der: 0.0 })
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Dual or float"))
    }
}

use pyo3::prelude::*;
use super::core::Array;
use crate::vector::Vector;
use rayon::prelude::*;
use rand::Rng;

pub fn register_nn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Activations
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(leaky_relu, m)?)?;
    m.add_function(wrap_pyfunction!(gelu, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    
    // Loss
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, m)?)?;
    
    // Layers
    m.add_function(wrap_pyfunction!(batch_norm, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(dropout, m)?)?;
    
    // Initializers
    let init_mod = PyModule::new(m.py(), "initializers")?;
    register_initializers(&init_mod)?;
    m.add_submodule(&init_mod)?;
    pyo3::py_run!(m.py(), init_mod, "import sys; sys.modules['rmath.nn.initializers'] = init_mod");

    Ok(())
}

fn register_initializers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(glorot_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(glorot_normal, m)?)?;
    m.add_function(wrap_pyfunction!(he_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(he_normal, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (*shape))]
pub fn glorot_uniform(shape: Vec<usize>) -> Array {
    let fan_in = if shape.len() > 1 { shape[1] } else { shape[0] };
    let fan_out = shape[0];
    let limit = (6.0 / (fan_in as f64 + fan_out as f64)).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data = (0..n).map(|_| rng.gen_range(-limit..limit)).collect();
    Array::from_flat(data, shape)
}

#[pyfunction]
#[pyo3(signature = (*shape))]
pub fn glorot_normal(shape: Vec<usize>) -> Array {
    let fan_in = if shape.len() > 1 { shape[1] } else { shape[0] };
    let fan_out = shape[0];
    let std = (2.0 / (fan_in as f64 + fan_out as f64)).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data = (0..n).map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal) * std).collect();
    Array::from_flat(data, shape)
}

#[pyfunction]
#[pyo3(signature = (*shape))]
pub fn he_uniform(shape: Vec<usize>) -> Array {
    let fan_in = if shape.len() > 1 { shape[1] } else { shape[0] };
    let limit = (6.0 / fan_in as f64).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data = (0..n).map(|_| rng.gen_range(-limit..limit)).collect();
    Array::from_flat(data, shape)
}

#[pyfunction]
#[pyo3(signature = (*shape))]
pub fn he_normal(shape: Vec<usize>) -> Array {
    let fan_in = if shape.len() > 1 { shape[1] } else { shape[0] };
    let std = (2.0 / fan_in as f64).sqrt();
    let n: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data = (0..n).map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal) * std).collect();
    Array::from_flat(data, shape)
}

#[pymethods]
impl Array {

    // ── Activations ───────────────────────────────────────────────────────

    /// Apply the Sigmoid activation function element-wise.
    ///
    /// σ(x) = 1 / (1 + e⁻ˣ)
    pub fn sigmoid(&self) -> Self {
        self.map_elements(|x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn sigmoid_deriv(&self) -> Self {
        self.map_elements(|x| {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s)
        })
    }

    /// Apply the Rectified Linear Unit (ReLU) activation function element-wise.
    ///
    /// f(x) = max(0, x)
    pub fn relu(&self) -> Self { self.map_elements(|x| x.max(0.0)) }

    /// Derivative of ReLU.
    pub fn relu_deriv(&self) -> Self { self.map_elements(|x| if x > 0.0 { 1.0 } else { 0.0 }) }

    pub fn leaky_relu(&self, alpha: f64) -> Self {
        self.map_elements(|x| if x >= 0.0 { x } else { alpha * x })
    }

    pub fn leaky_relu_deriv(&self, alpha: f64) -> Self {
        self.map_elements(|x| if x >= 0.0 { 1.0 } else { alpha })
    }

    pub fn elu(&self, alpha: f64) -> Self {
        self.map_elements(|x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) })
    }

    pub fn selu(&self) -> Self {
        const ALPHA: f64 = 1.6732632423543772;
        const SCALE: f64 = 1.0507009873554805;
        self.map_elements(|x| SCALE * if x >= 0.0 { x } else { ALPHA * (x.exp() - 1.0) })
    }

    /// Apply the Gaussian Error Linear Unit (GELU) activation function.
    ///
    /// This uses the standard approximation for computational efficiency.
    pub fn gelu(&self) -> Self {
        // Approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))
        self.map_elements(|x| {
            let c = (2.0f64 / std::f64::consts::PI).sqrt();
            x * 0.5 * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
        })
    }

    pub fn swish(&self) -> Self {
        self.map_elements(|x| x / (1.0 + (-x).exp()))
    }

    pub fn mish(&self) -> Self {
        self.map_elements(|x| x * ((1.0 + x.exp()).ln()).tanh())
    }

    pub fn softplus(&self) -> Self {
        self.map_elements(|x| (1.0 + x.exp()).ln())
    }

    /// Softmax along axis=1 (row-wise, standard for classification)
    pub fn softmax(&self) -> Self {
        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        let out: Vec<f64> = (0..r).into_par_iter().map(|i| {
            let row = &d[i*c..(i+1)*c];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            exps.into_iter().map(move |e| e / sum).collect::<Vec<f64>>()
        }).flatten().collect();
        Self::from_flat(out, vec![r, c])
    }

    pub fn log_softmax(&self) -> Self {
        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        let out: Vec<f64> = (0..r).into_par_iter().map(|i| {
            let row = &d[i*c..(i+1)*c];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_sum = sum_exp.ln();
            row.iter().map(move |&x| (x - max_val) - log_sum).collect::<Vec<f64>>()
        }).flatten().collect();
        Self::from_flat(out, vec![r, c])
    }

    pub fn hardswish(&self) -> Self {
        self.map_elements(|x| x * ((x + 3.0).clamp(0.0, 6.0) / 6.0))
    }

    // ── Loss functions ────────────────────────────────────────────────────

    /// Compute the Mean Squared Error (MSE) loss between this array and Target.
    pub fn mse_loss(&self, target: &Array) -> PyResult<f64> {
        if self.shape != target.shape {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch"));
        }
        let n = self.len() as f64;
        let loss: f64 = self.data().par_iter().zip(target.data().par_iter())
            .map(|(&p, &t)| (p - t).powi(2)).sum::<f64>() / n;
        Ok(loss)
    }

    /// Mean Absolute Error
    pub fn mae_loss(&self, target: &Array) -> PyResult<f64> {
        if self.shape != target.shape {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch"));
        }
        let n = self.len() as f64;
        let loss: f64 = self.data().par_iter().zip(target.data().par_iter())
            .map(|(&p, &t)| (p - t).abs()).sum::<f64>() / n;
        Ok(loss)
    }

    /// Huber loss (smooth L1)
    pub fn huber_loss(&self, target: &Array, delta: f64) -> PyResult<f64> {
        if self.shape != target.shape {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch"));
        }
        let n = self.len() as f64;
        let loss: f64 = self.data().par_iter().zip(target.data().par_iter()).map(|(&p, &t)| {
            let e = (p - t).abs();
            if e <= delta { 0.5 * e * e } else { delta * (e - 0.5 * delta) }
        }).sum::<f64>() / n;
        Ok(loss)
    }

    /// Binary cross-entropy (predictions must be probabilities)
    pub fn binary_cross_entropy(&self, target: &Array) -> PyResult<f64> {
        if self.shape != target.shape {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch"));
        }
        let eps = 1e-15;
        let n = self.len() as f64;
        let loss: f64 = self.data().par_iter().zip(target.data().par_iter()).map(|(&p, &t)| {
            let p = p.clamp(eps, 1.0 - eps);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        }).sum::<f64>() / n;
        Ok(loss)
    }

    /// Compute the Cross-Entropy loss.
    ///
    /// Requires that this array contains log-probabilities (e.g., from `log_softmax`).
    /// Labels should be a list of integer class indices.
    pub fn cross_entropy_loss(&self, labels: Vec<usize>) -> PyResult<f64> {
        let r = self.nrows();
        if labels.len() != r {
            return Err(pyo3::exceptions::PyValueError::new_err("labels len mismatch"));
        }
        let c = self.ncols();
        let d = self.data();
        // self should be log_softmax output
        let loss: f64 = labels.iter().enumerate()
            .map(|(i, &lbl)| -d[i*c + lbl]).sum::<f64>() / r as f64;
        Ok(loss)
    }

    // ── Normalization ─────────────────────────────────────────────────────

    /// Batch normalization (inference mode): (x - mu) / sigma * gamma + beta
    pub fn batch_norm(&self, mu: &Vector, sigma: &Vector, gamma: &Vector, beta: &Vector) -> PyResult<Self> {
        let c = self.ncols();
        if mu.len_internal() != c || sigma.len_internal() != c
            || gamma.len_internal() != c || beta.len_internal() != c {
            return Err(pyo3::exceptions::PyValueError::new_err("Param dim mismatch"));
        }
        let (r, _) = (self.nrows(), c);
        let d = self.data();
        let data: Vec<f64> = mu.with_slice(|ms| {
            sigma.with_slice(|ss| {
                gamma.with_slice(|gs| {
                    beta.with_slice(|bs| {
                        (0..r*c).into_par_iter().map(|k| {
                            let j = k % c;
                            gs[j] * (d[k] - ms[j]) / ss[j] + bs[j]
                        }).collect()
                    })
                })
            })
        });
        Ok(Self::from_flat(data, vec![r, c]))
    }

    /// Layer normalization (normalize each row independently)
    pub fn layer_norm(&self, eps: f64) -> Self {
        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        let out: Vec<f64> = (0..r).into_par_iter().map(|i| {
            let row = &d[i*c..(i+1)*c];
            let mu: f64 = row.iter().sum::<f64>() / c as f64;
            let var: f64 = row.iter().map(|&x| (x-mu).powi(2)).sum::<f64>() / c as f64;
            let std = (var + eps).sqrt();
            row.iter().map(move |&x| (x - mu) / std).collect::<Vec<f64>>()
        }).flatten().collect();
        Self::from_flat(out, vec![r, c])
    }

    // ── Dropout ───────────────────────────────────────────────────────────

    /// Apply dropout during training.
    ///
    /// Zeroes out elements with probability `p` and scales remaining elements by `1/(1-p)`.
    pub fn dropout(&self, p: f64) -> PyResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(pyo3::exceptions::PyValueError::new_err("p must be in [0,1)"));
        }
        use rand::Rng;
        let scale = 1.0 / (1.0 - p);
        let data: Vec<f64> = self.data().par_iter()
            .map_init(rand::thread_rng, |rng, &x| if rng.r#gen::<f64>() > p { x * scale } else { 0.0 })
            .collect();
        Ok(Self::from_flat(data, self.shape.clone()))
    }

    // ── Padding ───────────────────────────────────────────────────────────

    /// Pad rows/cols with a constant value: (top, bottom, left, right)
    pub fn pad(&self, top: usize, bottom: usize, left: usize, right: usize, val: f64) -> Self {
        let (r, c) = (self.nrows(), self.ncols());
        let new_r = r + top + bottom;
        let new_c = c + left + right;
        let d = self.data();
        let mut out = vec![val; new_r * new_c];
        for i in 0..r {
            for j in 0..c {
                out[(i+top)*new_c + (j+left)] = d[i*c+j];
            }
        }
        Self::from_flat(out, vec![new_r, new_c])
    }

    // ── Pooling (for 2-D feature maps) ────────────────────────────────────

    /// Perform 2D Max Pooling.
    ///
    /// Args:
    ///     kernel_size: Size of the square pooling window.
    pub fn max_pool2d(&self, kernel_size: usize) -> PyResult<Self> {
        let (r, c) = (self.nrows(), self.ncols());
        if r % kernel_size != 0 || c % kernel_size != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows and cols must be divisible by kernel_size"));
        }
        let out_r = r / kernel_size;
        let out_c = c / kernel_size;
        let d = self.data();
        let out: Vec<f64> = (0..out_r).into_par_iter().map(|oi| {
            let mut row = Vec::with_capacity(out_c);
            for oj in 0..out_c {
                let mut m = f64::NEG_INFINITY;
                for ki in 0..kernel_size { for kj in 0..kernel_size {
                    let v = d[(oi*kernel_size+ki)*c + (oj*kernel_size+kj)];
                    if v > m { m = v; }
                }}
                row.push(m);
            }
            row
        }).flatten().collect();
        Ok(Self::from_flat(out, vec![out_r, out_c]))
    }

    /// Average pooling with kernel_size × kernel_size, stride = kernel_size
    pub fn avg_pool2d(&self, kernel_size: usize) -> PyResult<Self> {
        let (r, c) = (self.nrows(), self.ncols());
        if r % kernel_size != 0 || c % kernel_size != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rows and cols must be divisible by kernel_size"));
        }
        let out_r = r / kernel_size;
        let out_c = c / kernel_size;
        let k2 = (kernel_size * kernel_size) as f64;
        let d = self.data();
        let out: Vec<f64> = (0..out_r).into_par_iter().map(|oi| {
            let mut row = Vec::with_capacity(out_c);
            for oj in 0..out_c {
                let mut s = 0.0;
                for ki in 0..kernel_size { for kj in 0..kernel_size {
                    s += d[(oi*kernel_size+ki)*c + (oj*kernel_size+kj)];
                }}
                row.push(s / k2);
            }
            row
        }).flatten().collect();
        Ok(Self::from_flat(out, vec![out_r, out_c]))
    }

    // ── Gradient helpers ─────────────────────────────────────────────────

    /// MSE gradient w.r.t. predictions: 2(pred - target) / n
    pub fn mse_grad(&self, target: &Array) -> PyResult<Self> {
        if self.shape != target.shape {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch"));
        }
        let n = self.len() as f64;
        let data: Vec<f64> = self.data().par_iter().zip(target.data().par_iter())
            .map(|(&p, &t)| 2.0 * (p - t) / n).collect();
        Ok(Self::from_flat(data, self.shape.clone()))
    }

    /// Softmax + cross-entropy combined gradient: softmax(x) - one_hot(labels)
    pub fn softmax_ce_grad(&self, labels: Vec<usize>) -> PyResult<Self> {
        let r = self.nrows();
        if labels.len() != r {
            return Err(pyo3::exceptions::PyValueError::new_err("labels len mismatch"));
        }
        let mut sm = self.softmax();
        sm.make_owned();
        let c = sm.ncols();
        match &mut sm.storage {
            super::core::ArrayStorage::Inline(d, _) => {
                for (i, &lbl) in labels.iter().enumerate() { d[i*c+lbl] -= 1.0; }
            }
            super::core::ArrayStorage::Heap(arc) => {
                let data = std::sync::Arc::get_mut(arc).unwrap();
                for (i, &lbl) in labels.iter().enumerate() { data[i*c+lbl] -= 1.0; }
            }
        }
        Ok(sm)
    }
}

// ── Functional API ───────────────────────────────────────────────────────────

#[pyfunction]
pub fn relu(a: &Array) -> Array { a.relu() }

#[pyfunction]
pub fn sigmoid(a: &Array) -> Array { a.sigmoid() }

#[pyfunction]
pub fn leaky_relu(a: &Array, alpha: f64) -> Array { a.leaky_relu(alpha) }

#[pyfunction]
pub fn gelu(a: &Array) -> Array { a.gelu() }

#[pyfunction]
pub fn softmax(a: &Array) -> Array { a.softmax() }

#[pyfunction]
pub fn mse_loss(a: &Array, target: &Array) -> PyResult<f64> { a.mse_loss(target) }

#[pyfunction]
pub fn cross_entropy_loss(a: &Array, labels: Vec<usize>) -> PyResult<f64> { a.cross_entropy_loss(labels) }

#[pyfunction]
pub fn batch_norm(a: &Array, mu: &Vector, sigma: &Vector, gamma: &Vector, beta: &Vector) -> PyResult<Array> {
    a.batch_norm(mu, sigma, gamma, beta)
}

#[pyfunction]
pub fn layer_norm(a: &Array, eps: f64) -> Array { a.layer_norm(eps) }

#[pyfunction]
pub fn dropout(a: &Array, p: f64) -> PyResult<Array> { a.dropout(p) }
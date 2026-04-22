use pyo3::prelude::*;
use rayon::prelude::*;
use super::autograd::Tensor;
use super::core::Array;

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Supports optional momentum for faster convergence.
///
/// Example:
///     >>> optimizer = ra.SGD(model.parameters(), lr=0.01, momentum=0.9)
///     >>> optimizer.step()
#[pyclass(module = "rmath.array")]
pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub momentum: f64,
    pub velocities: Vec<Option<Array>>,
}

#[pymethods]
impl SGD {
    #[new]
    #[pyo3(signature = (params, lr, momentum = 0.0))]
    pub fn new(params: Vec<Tensor>, lr: f64, momentum: f64) -> Self {
        let n = params.len();
        SGD { 
            params, 
            lr, 
            momentum, 
            velocities: vec![None; n] 
        }
    }

    pub fn step(&mut self) -> PyResult<()> {
        let lr = self.lr;
        let momentum = self.momentum;

        for i in 0..self.params.len() {
            let param = &mut self.params[i];
            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let mut p_lock = param.data_ptr.write().unwrap();
                let p_data = p_lock.storage_slice_mut();
                let g_data = grad.data(); // Contiguous copy if needed
                
                if momentum > 0.0 {
                    if self.velocities[i].is_none() {
                        self.velocities[i] = Some(Array::zeros_internal(&grad.shape));
                    }
                    let v_arr = self.velocities[i].as_mut().unwrap();
                    let v_data = v_arr.storage_slice_mut();

                    p_data.par_iter_mut().zip(v_data.par_iter_mut()).zip(g_data.par_iter()).for_each(|((p, v), &g)| {
                        *v = momentum * (*v) + lr * g;
                        *p -= *v;
                    });
                } else {
                    p_data.par_iter_mut().zip(g_data.par_iter()).for_each(|(p, &g)| {
                        *p -= lr * g;
                    });
                }
            }
        }
        Ok(())
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

/// Adam Optimizer: The industry standard for deep learning.
///
/// Implements the Adam algorithm with bias correction and fused parameter updates.
///
/// Example:
///     >>> optimizer = ra.Adam(model.parameters(), lr=1e-3)
///     >>> optimizer.step()
#[pyclass(module = "rmath.array")]
pub struct Adam {
    params: Vec<Tensor>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<Array>,
    v: Vec<Array>,
    t: usize,
}

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8))]
    pub fn new(params: Vec<Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        let m = params.iter().map(|p| Array::zeros_internal(&p.data().shape())).collect();
        let v = params.iter().map(|p| Array::zeros_internal(&p.data().shape())).collect();
        Adam { params, lr, beta1, beta2, eps, m, v, t: 0 }
    }

    pub fn step(&mut self) -> PyResult<()> {
        self.t += 1;
        let t = self.t as i32;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let lr = self.lr;

        // Correct bias
        let bias_correction1 = 1.0 - beta1.powi(t);
        let bias_correction2 = 1.0 - beta2.powi(t);
        let lr_t = lr * bias_correction2.sqrt() / bias_correction1;

        for i in 0..self.params.len() {
            let param = &mut self.params[i];
            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let mut p_lock = param.data_ptr.write().unwrap();
                let p_data = p_lock.storage_slice_mut();
                let g_data = grad.data();
                
                let m_data = self.m[i].storage_slice_mut();
                let v_data = self.v[i].storage_slice_mut();

                // ONE PASS FUSION: m, v, and param update in one loop
                p_data.par_iter_mut()
                    .zip(m_data.par_iter_mut())
                    .zip(v_data.par_iter_mut())
                    .zip(g_data.par_iter())
                    .for_each(|(((p, m), v), &g)| {
                        // m = beta1 * m + (1 - beta1) * g
                        *m = beta1 * (*m) + (1.0 - beta1) * g;
                        // v = beta2 * v + (1 - beta2) * g^2
                        *v = beta2 * (*v) + (1.0 - beta2) * g * g;
                        // p = p - lr_t * m / (sqrt(v) + eps)
                        *p -= lr_t * (*m) / ((*v).sqrt() + eps);
                    });
            }
        }
        Ok(())
    }

    pub fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

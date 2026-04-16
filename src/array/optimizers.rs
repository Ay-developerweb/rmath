use pyo3::prelude::*;
use super::autograd::Tensor;
use super::core::Array;

#[pyclass(module = "rmath")]
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
        for i in 0..self.params.len() {
            let param = &mut self.params[i];
            if let Some(grad) = param.grad() {
                let velocity = if self.momentum > 0.0 {
                    let v = match &self.velocities[i] {
                        Some(prev_v) => {
                            // v = momentum * prev_v + lr * grad
                            let term1 = prev_v.map_elements(|x| x * self.momentum);
                            let term2 = grad.map_elements(|x| x * self.lr);
                            term1.add_array(&term2)?
                        }
                        None => grad.map_elements(|x| x * self.lr),
                    };
                    self.velocities[i] = Some(v.clone());
                    v
                } else {
                    grad.map_elements(|x| x * self.lr)
                };

                let new_data = param.data().sub_array(&velocity)?;
                param.update_data(new_data);
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

/// Adam Optimizer: The industry standard for deep learning
#[pyclass(module = "rmath")]
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
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.t as i32)).sqrt() / (1.0 - self.beta1.powi(self.t as i32));

        for i in 0..self.params.len() {
            let param = &mut self.params[i];
            if let Some(grad) = param.grad() {
                // m = beta1 * m + (1 - beta1) * grad
                let m_part1 = self.m[i].map_elements(|x| x * self.beta1);
                let m_part2 = grad.map_elements(|x| x * (1.0 - self.beta1));
                self.m[i] = m_part1.add_array(&m_part2)?;

                // v = beta2 * v + (1 - beta2) * grad^2
                let v_part1 = self.v[i].map_elements(|x| x * self.beta2);
                let grad_sq = grad.map_elements(|x| x * x);
                let v_part2 = grad_sq.map_elements(|x| x * (1.0 - self.beta2));
                self.v[i] = v_part1.add_array(&v_part2)?;

                // weight = weight - lr_t * m / (sqrt(v) + eps)
                let m = &self.m[i];
                let v = &self.v[i];
                let denom = v.map_elements(|x| x.sqrt() + self.eps);
                
                // This is a bit complex for our current elementwise helpers, 
                // but we can map it manually for now.
                let update_data: Vec<f64> = m.data().iter().zip(denom.data().iter())
                    .map(|(&mi, &di)| mi / di * lr_t).collect();
                let update = Array::from_flat(update_data, m.shape.clone());

                let new_data = param.data().sub_array(&update)?;
                param.update_data(new_data);
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

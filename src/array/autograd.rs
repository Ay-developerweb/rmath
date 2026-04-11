use pyo3::prelude::*;
use super::core::Array;
use std::sync::{Arc, Mutex, RwLock};

/// OpType tracks the operation that created this tensor for the backward pass.
#[derive(Clone)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    MatMul,
    Sigmoid,
    ReLU,
    None, // For leaf nodes
}

/// A node in the computation graph.
pub struct GraphNode {
    pub op: OpType,
    pub inputs: Vec<Tensor>,
}

/// Tensor wraps rmath::Array with Autograd capabilities.
/// Shared state (RwLock) ensures updates are reflected across all clones.
#[pyclass]
#[derive(Clone)]
pub struct Tensor {
    pub data_ptr: Arc<RwLock<Array>>,
    pub grad: Arc<Mutex<Option<Array>>>,
    #[pyo3(get, set)]
    pub requires_grad: bool,
    pub node: Option<Arc<GraphNode>>,
}

#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (data, requires_grad = false))]
    pub fn new(data: Array, requires_grad: bool) -> Self {
        Tensor {
            data_ptr: Arc::new(RwLock::new(data)),
            grad: Arc::new(Mutex::new(None)),
            requires_grad,
            node: None,
        }
    }

    /// Access the underlying Array data
    #[getter]
    pub fn data(&self) -> Array {
        self.data_ptr.read().unwrap().clone()
    }

    /// Access the gradient as an Array (returns None if not computed)
    #[getter]
    pub fn grad(&self) -> Option<Array> {
        self.grad.lock().unwrap().clone()
    }

    /// Zero the gradient
    pub fn zero_grad(&self) {
        let mut g = self.grad.lock().unwrap();
        *g = None;
    }

    /// Update the underlying data (used by optimizers)
    pub fn update_data(&mut self, new_data: Array) {
        let mut data = self.data_ptr.write().unwrap();
        *data = new_data;
    }

    /// Topological sort to get all nodes in execution order
    pub fn build_topo(&self) -> Vec<Tensor> {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        fn walk(tensor: &Tensor, topo: &mut Vec<Tensor>, visited: &mut std::collections::HashSet<usize>) {
            let ptr = tensor as *const Tensor as usize;
            if visited.contains(&ptr) { return; }
            visited.insert(ptr);
            
            if let Some(node) = &tensor.node {
                for input in &node.inputs {
                    walk(input, topo, visited);
                }
            }
            topo.push(tensor.clone());
        }
        
        walk(self, &mut topo, &mut visited);
        topo
    }

    /// Iterative backward implementation using topological sort
    pub fn backward(&self) {
        if !self.requires_grad { return; }
        
        // Ensure root gradient is 1.0
        {
            let mut g = self.grad.lock().unwrap();
            if g.is_none() {
                let data = self.data();
                *g = Some(Array::full_internal(&data.shape(), 1.0));
            }
        }
        
        // Traverse in reverse topological order (sink to sources)
        let topo = self.build_topo();
        for tensor in topo.iter().rev() {
            tensor.backward_step();
        }
    }

    pub fn __add__(&self, other: &Tensor) -> PyResult<Self> {
        let res_data = self.data().add_array(&other.data())?;
        let mut res = Tensor::new(res_data, self.requires_grad || other.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Add, inputs: vec![self.clone(), other.clone()] }));
        }
        Ok(res)
    }

    pub fn __sub__(&self, other: &Tensor) -> PyResult<Self> {
        let res_data = self.data().sub_array(&other.data())?;
        let mut res = Tensor::new(res_data, self.requires_grad || other.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Sub, inputs: vec![self.clone(), other.clone()] }));
        }
        Ok(res)
    }

    pub fn __mul__(&self, other: &Tensor) -> PyResult<Self> {
        let res_data = self.data().mul_array_elementwise(&other.data())?;
        let mut res = Tensor::new(res_data, self.requires_grad || other.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Mul, inputs: vec![self.clone(), other.clone()] }));
        }
        Ok(res)
    }

    pub fn __matmul__(&self, other: &Tensor) -> PyResult<Self> {
        let res_data = self.data().matmul_array(&other.data());
        let mut res = Tensor::new(res_data, self.requires_grad || other.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::MatMul, inputs: vec![self.clone(), other.clone()] }));
        }
        Ok(res)
    }

    pub fn sigmoid(&self) -> Self {
        let res_data = self.data().sigmoid();
        let mut res = Tensor::new(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Sigmoid, inputs: vec![self.clone()] }));
        }
        res
    }

    pub fn relu(&self) -> Self {
        let res_data = self.data().relu();
        let mut res = Tensor::new(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::ReLU, inputs: vec![self.clone()] }));
        }
        res
    }

    pub fn __repr__(&self) -> String {
        let op_name = match &self.node {
            Some(n) => match n.op {
                OpType::Add => "Add",
                OpType::Sub => "Sub",
                OpType::Mul => "Mul",
                OpType::MatMul => "MatMul",
                OpType::Sigmoid => "Sigmoid",
                OpType::ReLU => "ReLU",
                OpType::None => "None",
            },
            None => "None",
        };
        format!("Tensor(data={}, grad_fn={}, requires_grad={})", 
            self.data().__repr__(), op_name, self.requires_grad)
    }
}

impl Tensor {
    fn backward_step(&self) {
        let node = match &self.node {
            Some(n) => n,
            None => return,
        };

        let grad = match self.grad.lock().unwrap().clone() {
            Some(g) => g,
            None => return,
        };

        match node.op {
            OpType::Add => {
                for input in &node.inputs {
                    if input.requires_grad {
                        input.accumulate_grad(&grad);
                    }
                }
            }
            OpType::Sub => {
                if node.inputs[0].requires_grad {
                    node.inputs[0].accumulate_grad(&grad);
                }
                if node.inputs[1].requires_grad {
                    let neg_grad = grad.map_elements(|x| -x);
                    node.inputs[1].accumulate_grad(&neg_grad);
                }
            }
            OpType::Mul => {
                if node.inputs[0].requires_grad {
                    let ig = grad.mul_array_elementwise(&node.inputs[1].data()).unwrap();
                    node.inputs[0].accumulate_grad(&ig);
                }
                if node.inputs[1].requires_grad {
                    let ig = grad.mul_array_elementwise(&node.inputs[0].data()).unwrap();
                    node.inputs[1].accumulate_grad(&ig);
                }
            }
            OpType::MatMul => {
                let a = &node.inputs[0];
                let b = &node.inputs[1];
                if a.requires_grad {
                    let ig = grad.matmul_array(&b.data().transpose_internal());
                    a.accumulate_grad(&ig);
                }
                if b.requires_grad {
                    let ig = a.data().transpose_internal().matmul_array(&grad);
                    b.accumulate_grad(&ig);
                }
            }
            OpType::Sigmoid => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let deriv = input.data().sigmoid_deriv();
                    let ig = grad.mul_array_elementwise(&deriv).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            OpType::ReLU => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let deriv = input.data().relu_deriv();
                    let ig = grad.mul_array_elementwise(&deriv).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            _ => {}
        }
    }

    fn accumulate_grad(&self, grad: &Array) {
        let mut g = self.grad.lock().unwrap();
        *g = Some(match g.as_ref() {
            Some(existing) => existing.add_array(grad).unwrap(),
            None => grad.clone(),
        });
    }
}

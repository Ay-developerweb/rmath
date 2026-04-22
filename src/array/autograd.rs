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
    Div,
    Sum,
    Exp,
    Log,
    Tanh,
    Neg,
    Abs,
    Mean,
    None, // For leaf nodes
}

/// A node in the computation graph.
pub struct GraphNode {
    pub op: OpType,
    pub inputs: Vec<Tensor>,
}

/// Tensor wraps rmath::Array with Autograd capabilities.
/// 
/// `Tensor` supports automatic differentiation by tracking operations 
/// and building a dynamic computation graph. This allows for seamless 
/// gradient computation for deep learning and optimization.
///
/// Examples:
///     >>> import rmath.array as ra
///     >>> x = ra.Tensor([1.0, 2.0], requires_grad=True)
///     >>> y = x * 2 + 5
///     >>> z = y.sum()
///     >>> z.backward()
///     >>> x.grad
///     Array([2.0, 2.0])
#[pyclass(module = "rmath.array")]
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
    #[pyo3(signature = (data, requires_grad = false, **kwargs))]
    pub fn new(data: Bound<'_, PyAny>, requires_grad: bool, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut arr = Tensor::extract_array_from_any(&data)?;
        if let Some(kw) = kwargs {
            if let Ok(Some(dt)) = kw.get_item("dtype") {
                if let Ok(dt_str) = dt.extract::<String>() {
                    arr = arr.apply_dtype(Some(&dt_str))?;
                }
            }
        }
        Ok(Tensor {
            data_ptr: Arc::new(RwLock::new(arr)),
            grad: Arc::new(Mutex::new(None)),
            requires_grad,
            node: None,
        })
    }

    /// Internal wrapper to instantiate Tensor with an existing array safely
    #[staticmethod]
    fn from_array(arr: Array, requires_grad: bool) -> Self {
        Tensor {
            data_ptr: Arc::new(RwLock::new(arr)),
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

    #[setter]
    pub fn set_grad(&self, grad: Option<Array>) {
        let mut g = self.grad.lock().unwrap();
        *g = grad;
    }

    /// Zero the gradient
    pub fn zero_grad(&self) {
        let mut g = self.grad.lock().unwrap();
        *g = None;
    }

    /// Set requires_grad in place (Torch alias)
    #[pyo3(name = "requires_grad_")]
    pub fn requires_grad_(&mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self.clone()
    }

    /// Update the underlying data (used by optimizers)
    pub fn update_data(&mut self, new_data: Array) {
        let mut data = self.data_ptr.write().unwrap();
        *data = new_data;
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.data_ptr.read().unwrap().shape.clone()
    }

    #[getter]
    pub fn dtype(&self) -> String {
        self.data_ptr.read().unwrap().dtype.clone()
    }

    #[getter]
    #[pyo3(name = "ndim")]
    pub fn ndim_py(&self) -> usize {
        self.data_ptr.read().unwrap().shape.len()
    }

    pub fn __getitem__(&self, py: Python<'_>, index: Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.data().__getitem__(py, index)
    }

    /// Topological sort to get all nodes in execution order
    pub fn build_topo(&self) -> Vec<Tensor> {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        fn walk(tensor: &Tensor, topo: &mut Vec<Tensor>, visited: &mut std::collections::HashSet<usize>) {
            let ptr = Arc::as_ptr(&tensor.data_ptr) as usize;
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

    /// Compute gradients using backpropagation.
    ///
    /// This will traverse the computation graph from this tensor back to the leaves,
    /// accumulating gradients in each tensor that has `requires_grad=True`.
    ///
    /// Example:
    ///     >>> loss = (model(x) - y).pow(2).mean()
    ///     >>> loss.backward()
    pub fn backward(&self) {
        if !self.requires_grad { return; }
        
        // Ensure root gradient is 1.0
        {
            let mut g = self.grad.lock().unwrap();
            if g.is_none() {
                let shape = self.data_ptr.read().unwrap().shape.clone();
                *g = Some(Array::full_internal(&shape, 1.0));
            }
        }
        
        // Traverse in reverse topological order (sink to sources)
        let topo = self.build_topo();
        for tensor in topo.iter().rev() {
            tensor.backward_step();
        }
    }

    // ── Arithmetic Operators ─────────────────────────────────────────────

    pub fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        let other_tensor = Self::extract_tensor(other)?;
        let self_data = self.data_ptr.read().unwrap();
        let other_data = other_tensor.data_ptr.read().unwrap();
        let res_data = self_data.add_array(&other_data)?;
        drop(self_data); drop(other_data);
        let mut res = Tensor::from_array(res_data, self.requires_grad || other_tensor.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Add, inputs: vec![self.clone(), other_tensor] }));
        }
        Ok(res)
    }

    pub fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    pub fn __sub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        let other_tensor = Self::extract_tensor(other)?;
        let self_data = self.data_ptr.read().unwrap();
        let other_data = other_tensor.data_ptr.read().unwrap();
        let res_data = self_data.sub_array(&other_data)?;
        drop(self_data); drop(other_data);
        let mut res = Tensor::from_array(res_data, self.requires_grad || other_tensor.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Sub, inputs: vec![self.clone(), other_tensor] }));
        }
        Ok(res)
    }

    pub fn __rsub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        let other_tensor = Self::extract_tensor(other)?;
        let other_data = other_tensor.data_ptr.read().unwrap();
        let self_data = self.data_ptr.read().unwrap();
        let res_data = other_data.sub_array(&self_data)?;
        drop(other_data); drop(self_data);
        let mut res = Tensor::from_array(res_data, self.requires_grad || other_tensor.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Sub, inputs: vec![other_tensor, self.clone()] }));
        }
        Ok(res)
    }

    pub fn __mul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        let other_tensor = Self::extract_tensor(other)?;
        let self_data = self.data_ptr.read().unwrap();
        let other_data = other_tensor.data_ptr.read().unwrap();
        let res_data = self_data.mul_array_elementwise(&other_data)?;
        drop(self_data); drop(other_data);
        let mut res = Tensor::from_array(res_data, self.requires_grad || other_tensor.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Mul, inputs: vec![self.clone(), other_tensor] }));
        }
        Ok(res)
    }

    pub fn __rmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    pub fn __truediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        let other_tensor = Self::extract_tensor(other)?;
        let self_data = self.data_ptr.read().unwrap();
        let other_data = other_tensor.data_ptr.read().unwrap();
        // Pure Rust elementwise division — no Python dispatch
        let res_data = self_data.div_array_elementwise(&other_data)?;
        drop(self_data); drop(other_data);
        let mut res = Tensor::from_array(res_data, self.requires_grad || other_tensor.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Div, inputs: vec![self.clone(), other_tensor] }));
        }
        Ok(res)
    }

    pub fn __rtruediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        let other_tensor = Self::extract_tensor(other)?;
        let other_data = other_tensor.data_ptr.read().unwrap();
        let self_data = self.data_ptr.read().unwrap();
        let res_data = other_data.div_array_elementwise(&self_data)?;
        drop(other_data); drop(self_data);
        let mut res = Tensor::from_array(res_data, self.requires_grad || other_tensor.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Div, inputs: vec![other_tensor, self.clone()] }));
        }
        Ok(res)
    }

    pub fn __matmul__(&self, other: &Tensor) -> PyResult<Self> {
        let self_data = self.data_ptr.read().unwrap();
        let other_data = other.data_ptr.read().unwrap();
        let res_data = self_data.matmul_array(&other_data);
        drop(self_data); drop(other_data);
        let mut res = Tensor::from_array(res_data, self.requires_grad || other.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::MatMul, inputs: vec![self.clone(), other.clone()] }));
        }
        Ok(res)
    }

    pub fn __neg__(&self) -> Self {
        let res_data = self.data_ptr.read().unwrap().map_elements(|x| -x);
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Neg, inputs: vec![self.clone()] }));
        }
        res
    }

    // ── Activation / Math Ops (with autograd) ────────────────────────────

    pub fn sigmoid(&self) -> Self {
        let res_data = self.data_ptr.read().unwrap().sigmoid();
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Sigmoid, inputs: vec![self.clone()] }));
        }
        res
    }

    pub fn relu(&self) -> Self {
        let res_data = self.data_ptr.read().unwrap().relu();
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::ReLU, inputs: vec![self.clone()] }));
        }
        res
    }

    /// Element-wise exponential with autograd support.
    pub fn exp(&self) -> Self {
        let res_data = self.data_ptr.read().unwrap().exp();
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Exp, inputs: vec![self.clone()] }));
        }
        res
    }

    /// Element-wise natural logarithm with autograd support.
    pub fn log(&self) -> Self {
        let res_data = self.data_ptr.read().unwrap().log();
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Log, inputs: vec![self.clone()] }));
        }
        res
    }

    /// Element-wise tanh with autograd support.
    pub fn tanh(&self) -> Self {
        let res_data = self.data_ptr.read().unwrap().tanh();
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Tanh, inputs: vec![self.clone()] }));
        }
        res
    }

    /// Element-wise absolute value with autograd support.
    pub fn abs(&self) -> Self {
        let res_data = self.data_ptr.read().unwrap().abs();
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Abs, inputs: vec![self.clone()] }));
        }
        res
    }

    // ── Reduction Ops ────────────────────────────────────────────────────

    /// Sum all elements, returning a scalar Tensor with autograd support.
    pub fn sum(&self) -> Self {
        let d = self.data_ptr.read().unwrap();
        let total: f64 = d.sum_all();
        drop(d);
        let res_data = Array::from_flat(vec![total], vec![1]);
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Sum, inputs: vec![self.clone()] }));
        }
        res
    }

    /// Mean of all elements, returning a scalar Tensor.
    pub fn mean(&self) -> Self {
        let d = self.data_ptr.read().unwrap();
        let n = d.len() as f64;
        let total: f64 = d.sum_all();
        drop(d);
        let res_data = Array::from_flat(vec![total / n], vec![1]);
        let mut res = Tensor::from_array(res_data, self.requires_grad);
        if res.requires_grad {
            res.node = Some(Arc::new(GraphNode { op: OpType::Mean, inputs: vec![self.clone()] }));
        }
        res
    }

    // ── Shape Ops ────────────────────────────────────────────────────────

    /// Reshape the tensor (no autograd graph node needed — shape-only change).
    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let reshaped = self.data_ptr.read().unwrap().reshape(new_shape)?;
        Ok(Tensor::from_array(reshaped, self.requires_grad))
    }

    /// Transpose the last two dimensions.
    pub fn transpose(&self) -> Self {
        let t = self.data_ptr.read().unwrap().transpose_internal();
        Tensor::from_array(t.to_contiguous(), self.requires_grad)
    }

    /// Alias for transpose (PyTorch-style).
    #[pyo3(name = "t")]
    pub fn t_py(&self) -> Self {
        self.transpose()
    }

    // ── Display ──────────────────────────────────────────────────────────

    pub fn __repr__(&self) -> String {
        let op_name = match &self.node {
            Some(n) => match n.op {
                OpType::Add => "Add",
                OpType::Sub => "Sub",
                OpType::Mul => "Mul",
                OpType::MatMul => "MatMul",
                OpType::Sigmoid => "Sigmoid",
                OpType::ReLU => "ReLU",
                OpType::Div => "Div",
                OpType::Sum => "Sum",
                OpType::Exp => "Exp",
                OpType::Log => "Log",
                OpType::Tanh => "Tanh",
                OpType::Neg => "Neg",
                OpType::Abs => "Abs",
                OpType::Mean => "Mean",
                OpType::None => "None",
            },
            None => "None",
        };
        let repr = self.data_ptr.read().unwrap().__repr__();
        format!("Tensor(data={}, grad_fn={}, requires_grad={})", 
            repr, op_name, self.requires_grad)
    }

    // ── Static Constructors ──────────────────────────────────────────────

    #[staticmethod]
    #[pyo3(signature = (*shape, requires_grad = false, **kwargs))]
    pub fn zeros(shape: Vec<usize>, requires_grad: bool, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut arr = Array::zeros_internal(&shape);
        if let Some(kw) = kwargs {
            if let Ok(Some(dt)) = kw.get_item("dtype") {
                arr = arr.apply_dtype(Some(&dt.extract::<String>()?))?;
            }
        }
        Ok(Tensor::from_array(arr, requires_grad))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, requires_grad = false, **kwargs))]
    pub fn ones(shape: Vec<usize>, requires_grad: bool, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut arr = Array::ones_internal(&shape);
        if let Some(kw) = kwargs {
            if let Ok(Some(dt)) = kw.get_item("dtype") {
                arr = arr.apply_dtype(Some(&dt.extract::<String>()?))?;
            }
        }
        Ok(Tensor::from_array(arr, requires_grad))
    }

    #[staticmethod]
    #[pyo3(signature = (*shape, requires_grad = false, **kwargs))]
    pub fn randn(shape: Vec<usize>, requires_grad: bool, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut arr = Array::randn(shape);
        if let Some(kw) = kwargs {
            if let Ok(Some(dt)) = kw.get_item("dtype") {
                arr = arr.apply_dtype(Some(&dt.extract::<String>()?))?;
            }
        }
        Ok(Tensor::from_array(arr, requires_grad))
    }
}

impl Tensor {
    fn extract_tensor(other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        if let Ok(t) = other.extract::<PyRef<Tensor>>() {
            Ok(t.clone())
        } else if let Ok(s) = other.extract::<f64>() {
            Ok(Tensor::from_array(Array::from_flat(vec![s], vec![1]), false))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Expected Tensor or scalar"))
        }
    }
    
    fn extract_array_from_any(data: &Bound<'_, PyAny>) -> PyResult<Array> {
        if let Ok(arr) = data.extract::<PyRef<Array>>() {
            Ok(arr.clone())
        } else {
            // Fallback to Array sequence parser matching
            Array::new(data.clone(), None)
        }
    }

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
                    let b_data = node.inputs[1].data_ptr.read().unwrap();
                    let ig = grad.mul_array_elementwise(&b_data).unwrap();
                    drop(b_data);
                    node.inputs[0].accumulate_grad(&ig);
                }
                if node.inputs[1].requires_grad {
                    let a_data = node.inputs[0].data_ptr.read().unwrap();
                    let ig = grad.mul_array_elementwise(&a_data).unwrap();
                    drop(a_data);
                    node.inputs[1].accumulate_grad(&ig);
                }
            }
            OpType::Div => {
                let a = &node.inputs[0];
                let b = &node.inputs[1];
                
                let mut a_grad_lock = a.grad.lock().unwrap();
                let mut b_grad_lock = b.grad.lock().unwrap();
                
                // Ensure buffers exist if needed
                if a.requires_grad && a_grad_lock.is_none() {
                    let shape = a.data_ptr.read().unwrap().shape.clone();
                    *a_grad_lock = Some(Array::zeros_internal(&shape));
                }
                if b.requires_grad && b_grad_lock.is_none() {
                    let shape = b.data_ptr.read().unwrap().shape.clone();
                    *b_grad_lock = Some(Array::zeros_internal(&shape));
                }
                
                let a_data = a.data_ptr.read().unwrap();
                let b_data = b.data_ptr.read().unwrap();

                // Get direct mutable access to the gradient buffers
                let da_acc = a_grad_lock.as_mut().map(|g| g.storage_slice_mut());
                let db_acc = b_grad_lock.as_mut().map(|g| g.storage_slice_mut());
                
                // Execute zero-allocation fused kernel
                a_data.div_backward_fused(&b_data, &grad, da_acc, db_acc);
            }
            OpType::MatMul => {
                let a = &node.inputs[0];
                let b = &node.inputs[1];
                if a.requires_grad {
                    let b_data = b.data_ptr.read().unwrap();
                    let ig = grad.matmul_array(&b_data.transpose_internal());
                    drop(b_data);
                    a.accumulate_grad(&ig);
                }
                if b.requires_grad {
                    let a_data = a.data_ptr.read().unwrap();
                    let ig = a_data.transpose_internal().matmul_array(&grad);
                    drop(a_data);
                    b.accumulate_grad(&ig);
                }
            }
            OpType::Sigmoid => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let d = input.data_ptr.read().unwrap();
                    let deriv = d.sigmoid_deriv();
                    drop(d);
                    let ig = grad.mul_array_elementwise(&deriv).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            OpType::ReLU => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let d = input.data_ptr.read().unwrap();
                    let deriv = d.relu_deriv();
                    drop(d);
                    let ig = grad.mul_array_elementwise(&deriv).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            OpType::Sum => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let grad_val = grad.data()[0];
                    let shape = input.data_ptr.read().unwrap().shape.clone();
                    let ig = Array::full_internal(&shape, grad_val);
                    input.accumulate_grad(&ig);
                }
            }
            OpType::Mean => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let n = input.data_ptr.read().unwrap().len() as f64;
                    let grad_val = grad.data()[0] / n;
                    let shape = input.data_ptr.read().unwrap().shape.clone();
                    let ig = Array::full_internal(&shape, grad_val);
                    input.accumulate_grad(&ig);
                }
            }
            OpType::Exp => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let d = input.data_ptr.read().unwrap();
                    let exp_x = d.exp();
                    drop(d);
                    let ig = grad.mul_array_elementwise(&exp_x).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            OpType::Log => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let d = input.data_ptr.read().unwrap();
                    let recip = d.map_elements(|x| 1.0 / x);
                    drop(d);
                    let ig = grad.mul_array_elementwise(&recip).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            OpType::Tanh => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let d = input.data_ptr.read().unwrap();
                    let tanh_x = d.tanh();
                    drop(d);
                    let deriv = tanh_x.map_elements(|t| 1.0 - t * t);
                    let ig = grad.mul_array_elementwise(&deriv).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            OpType::Neg => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let ig = grad.map_elements(|x| -x);
                    input.accumulate_grad(&ig);
                }
            }
            OpType::Abs => {
                let input = &node.inputs[0];
                if input.requires_grad {
                    let d = input.data_ptr.read().unwrap();
                    let sign = d.map_elements(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 });
                    drop(d);
                    let ig = grad.mul_array_elementwise(&sign).unwrap();
                    input.accumulate_grad(&ig);
                }
            }
            _ => {}
        }
    }

    fn accumulate_grad(&self, grad: &Array) {
        let mut g = self.grad.lock().unwrap();
        if let Some(existing) = g.as_mut() {
            existing.add_assign_array(grad).unwrap();
        } else {
            *g = Some(grad.clone());
        }
    }
}

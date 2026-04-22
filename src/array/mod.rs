// src/array/mod.rs
// Wires all array submodules and exposes register_array()

pub mod core;
pub mod ops;
pub mod linalg;
pub mod ml;
pub mod lazy;
pub mod io;
pub mod interop;
pub mod autograd;
pub mod optimizers;

pub use core::Array;
pub use autograd::Tensor;
pub use lazy::{LazyArray, MmapArray, ChunkIterator, MmapChunkIterator};

use pyo3::prelude::*;

pub fn register_array(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", "rmath.array — high-performance N-dimensional array engine.

This module provides the `Array` and `Tensor` classes, which are optimized 
for massively parallel numerical computing, deep learning, and GB-scale 
data processing.

Key Features:
    1. Parallel Core: Every element-wise operation (sin, exp, arithmetic) 
       automatically uses the Rayon thread pool for data > 8,192 elements.
    2. Tiered Storage:
       - Inline: Small arrays (<32 elements) use zero-allocation stack storage.
       - Heap: Shared memory via Arc-wrapped contiguous buffers.
       - Mmap: Memory-mapped files for processing datasets larger than RAM.
    3. Autograd Engine: The `Tensor` class provides a dynamic computational 
       graph with automatic differentiation for building neural networks.
    4. Interoperability: Zero-copy conversion to and from NumPy ndarrays.

Examples:
    >>> import rmath.array as ra
    >>> a = ra.randn(1000, 1000)
    >>> b = ra.ones(1000, 1000)
    >>> c = (a * b).exp().sum()
")?;

    let py = m.py();

    // Core Array class
    m.add_class::<Array>()?;
    m.add_class::<Tensor>()?;
    m.add_class::<optimizers::SGD>()?;
    m.add_class::<optimizers::Adam>()?;

    // Submodules
    let linalg_mod = PyModule::new(py, "linalg")?;
    linalg::register_linalg(&linalg_mod)?;
    m.add_submodule(&linalg_mod)?;
    
    let nn_mod = PyModule::new(py, "nn")?;
    ml::register_nn(&nn_mod)?;
    m.add_submodule(&nn_mod)?;

    // Lazy loading classes
    m.add_class::<LazyArray>()?;
    m.add_class::<MmapArray>()?;
    m.add_class::<ChunkIterator>()?;
    m.add_class::<MmapChunkIterator>()?;

    // Functional API
    m.add_function(wrap_pyfunction!(ops::arange, m)?)?;
    m.add_function(wrap_pyfunction!(ops::array_range, m)?)?;
    m.add_function(wrap_pyfunction!(ops::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ones, m)?)?;
    m.add_function(wrap_pyfunction!(ops::randn, m)?)?;
    m.add_function(wrap_pyfunction!(ops::rand_uniform, m)?)?;

    Ok(())
}
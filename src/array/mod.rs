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

    Ok(())
}
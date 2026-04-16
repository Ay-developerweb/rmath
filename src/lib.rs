#![allow(unexpected_cfgs)]
use pyo3::prelude::*;

mod array;
mod constants;
mod scalar;
mod stats;
mod vector;
mod geometry;
mod special;
mod signal;
mod calculus;

/// RMath: A high-performance, multi-threaded mathematical toolkit in Rust.
#[pymodule]
fn _rmath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // 1. Constants (rmath.constants)
    let constants_mod = PyModule::new(py, "constants")?;
    constants::register_constants(&constants_mod)?;
    m.add_submodule(&constants_mod)?;
    pyo3::py_run!(py, constants_mod, "import sys; sys.modules['rmath.constants'] = constants_mod");

    // 2. Scalar Operations (rmath.scalar)
    let scalar_mod = PyModule::new(py, "scalar")?;
    scalar::register_scalar(&scalar_mod)?;
    m.add_submodule(&scalar_mod)?;
    pyo3::py_run!(py, scalar_mod, "import sys; sys.modules['rmath.scalar'] = scalar_mod");

    // 3. Statistical Analysis (rmath.stats)
    let stats_mod = PyModule::new(py, "stats")?;
    stats::register_stats(&stats_mod)?;
    m.add_submodule(&stats_mod)?;
    pyo3::py_run!(py, stats_mod, "import sys; sys.modules['rmath.stats'] = stats_mod");

    // 4. Vectorized Core (rmath.vector)
    let vector_mod = PyModule::new(py, "vector")?;
    vector::register_vector(&vector_mod)?;
    m.add_submodule(&vector_mod)?;
    pyo3::py_run!(py, vector_mod, "import sys; sys.modules['rmath.vector'] = vector_mod");

    // 5. Array Core (rmath.array)
    let array_mod = PyModule::new(py, "array")?;
    array::register_array(&array_mod)?;
    m.add_submodule(&array_mod)?;
    pyo3::py_run!(py, array_mod, "import sys; sys.modules['rmath.array'] = array_mod");

    // 5.1 Linear Algebra (rmath.linalg)
    let linalg_mod = PyModule::new(py, "linalg")?;
    array::linalg::register_linalg(&linalg_mod)?;
    m.add_submodule(&linalg_mod)?;
    pyo3::py_run!(py, linalg_mod, "import sys; sys.modules['rmath.linalg'] = linalg_mod");

    // 5.2 Neural Networks (rmath.nn)
    let nn_mod = PyModule::new(py, "nn")?;
    array::ml::register_nn(&nn_mod)?;
    m.add_submodule(&nn_mod)?;
    pyo3::py_run!(py, nn_mod, "import sys; sys.modules['rmath.nn'] = nn_mod");

    // 6. Geometry (rmath.geometry)
    let geom_mod = PyModule::new(py, "geometry")?;
    geometry::register_geometry(py, &geom_mod)?;
    m.add_submodule(&geom_mod)?;
    pyo3::py_run!(py, geom_mod, "import sys; sys.modules['rmath.geometry'] = geom_mod");

    // 7. Special Functions (rmath.special)
    let special_mod = PyModule::new(py, "special")?;
    special::register_special(&special_mod)?;
    m.add_submodule(&special_mod)?;
    pyo3::py_run!(py, special_mod, "import sys; sys.modules['rmath.special'] = special_mod");

    // 8. Signal Processing (rmath.signal)
    let signal_mod = PyModule::new(py, "signal")?;
    signal::register_signal(py, &signal_mod)?;
    m.add_submodule(&signal_mod)?;
    pyo3::py_run!(py, signal_mod, "import sys; sys.modules['rmath.signal'] = signal_mod");

    // 10. Calculus (rmath.calculus)
    let calculus_mod = PyModule::new(py, "calculus")?;
    calculus::register(py, &calculus_mod)?;
    m.add_submodule(&calculus_mod)?;
    pyo3::py_run!(py, calculus_mod, "import sys; sys.modules['rmath.calculus'] = calculus_mod");

    // Root-level functions
    m.add_function(wrap_pyfunction!(scalar::loop_range, m)?)?;
    m.add_function(wrap_pyfunction!(vector_sum, m)?)?;
    m.add_function(wrap_pyfunction!(vector_mean, m)?)?;
    m.add_function(wrap_pyfunction!(vector_min, m)?)?;
    m.add_function(wrap_pyfunction!(vector_max, m)?)?;

    // Root-level class aliases
    m.add_class::<scalar::Scalar>()?;
    m.add_class::<vector::Vector>()?;
    m.add_class::<array::Array>()?;
    m.add_class::<array::autograd::Tensor>()?;
    m.add_class::<array::lazy::LazyArray>()?;

    Ok(())
}

#[pyfunction(name = "sum")]
fn vector_sum(_py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(vec) = v.extract::<PyRef<vector::Vector>>() {
        Ok(vec.sum())
    } else if let Ok(vec) = vector::Vector::py_new(v.clone()) {
        Ok(vec.sum())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Vector or Array compatible sequence"))
    }
}

#[pyfunction(name = "mean")]
fn vector_mean(_py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(vec) = v.extract::<PyRef<vector::Vector>>() {
        Ok(vec.mean())
    } else if let Ok(vec) = vector::Vector::py_new(v.clone()) {
        Ok(vec.mean())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Vector or Array compatible sequence"))
    }
}

#[pyfunction(name = "min")]
fn vector_min(_py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(vec) = v.extract::<PyRef<vector::Vector>>() {
        Ok(vec.min())
    } else if let Ok(vec) = vector::Vector::py_new(v.clone()) {
        Ok(vec.min())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Vector or Array compatible sequence"))
    }
}

#[pyfunction(name = "max")]
fn vector_max(_py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(vec) = v.extract::<PyRef<vector::Vector>>() {
        Ok(vec.max())
    } else if let Ok(vec) = vector::Vector::py_new(v.clone()) {
        Ok(vec.max())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Expected Vector or Array compatible sequence"))
    }
}

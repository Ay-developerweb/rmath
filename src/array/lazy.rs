use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use super::core::Array;
use rayon::prelude::*;

// ─── Ops for Fusion ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum FusedOp {
    Sin, Cos, Exp, Tanh, Abs, Sigmoid,
    AddScalar(f64),
    MulScalar(f64),
    PowScalar(f64),
}

// ─── LazyArray ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum LazySource {
    File { path: String, format: LazyFormat },
    Memory(Array),
}

#[derive(Clone, Debug)]
pub enum LazyFormat {
    RMath, Csv, Bin,
}

/// A deferred execution pipeline for array operations.
///
/// `LazyArray` records operations (sin, exp, arithmetic) and executes them 
/// in a single parallel pass to maximize cache locality and minimize memory overhead.
///
/// Examples:
///     >>> a = ra.randn(1000, 1000)
///     >>> # Record operations (fused into a single pass)
///     >>> lazy = a.lazy().sin().add(1.0).exp()
///     >>> # Execute the pipeline
///     >>> result = lazy.execute()
#[pyclass(module = "rmath.array")]
#[derive(Clone)]
pub struct LazyArray {
    pub source: LazySource,
    pub recipe: Vec<FusedOp>,
    pub shape:  Option<Vec<usize>>,
}

impl LazyArray {
    pub fn new_from_array(arr: Array) -> Self {
        let shape = Some(arr.shape.clone());
        LazyArray {
            source: LazySource::Memory(arr),
            recipe: Vec::new(),
            shape,
        }
    }

    pub fn apply_recipe(&self, x: f64) -> f64 {
        let mut val = x;
        for op in &self.recipe {
            match op {
                FusedOp::Sin => val = val.sin(),
                FusedOp::Cos => val = val.cos(),
                FusedOp::Exp => val = val.exp(),
                FusedOp::Tanh => val = val.tanh(),
                FusedOp::Abs => val = val.abs(),
                FusedOp::Sigmoid => val = 1.0 / (1.0 + (-val).exp()),
                FusedOp::AddScalar(s) => val += s,
                FusedOp::MulScalar(s) => val *= s,
                FusedOp::PowScalar(s) => val = val.powf(*s),
            }
        }
        val
    }
}

#[pymethods]
impl LazyArray {
    #[staticmethod]
    pub fn open(path: String) -> PyResult<Self> {
        let fmt = if path.ends_with(".rmath")     { LazyFormat::RMath }
                  else if path.ends_with(".csv")  { LazyFormat::Csv }
                  else if path.ends_with(".bin")  { LazyFormat::Bin }
                  else {
                      return Err(pyo3::exceptions::PyValueError::new_err(
                          "Unsupported format. Use .rmath / .csv / .bin"));
                  };
        Ok(LazyArray { 
            source: LazySource::File { path, format: fmt }, 
            recipe: Vec::new(),
            shape: None,
        })
    }

    #[staticmethod]
    pub fn open_bin(path: String, rows: usize, cols: usize) -> Self {
        LazyArray {
            source: LazySource::File { path, format: LazyFormat::Bin },
            recipe: Vec::new(),
            shape: Some(vec![rows, cols]),
        }
    }

    pub fn peek(&mut self) -> PyResult<Vec<usize>> {
        if let Some(s) = &self.shape { return Ok(s.clone()); }
        
        match &self.source {
            LazySource::Memory(arr) => {
                let s = arr.shape.clone();
                self.shape = Some(s.clone());
                Ok(s)
            }
            LazySource::File { path, format } => {
                let shape = match format {
                    LazyFormat::RMath => rmath_read_header(path)?,
                    LazyFormat::Csv => {
                        let (r, c) = csv_count_shape(path)?;
                        vec![r, c]
                    }
                    LazyFormat::Bin => return Err(pyo3::exceptions::PyValueError::new_err(
                        ".bin files need explicit shape — use LazyArray.open_bin(path, rows, cols)")),
                };
                self.shape = Some(shape.clone());
                Ok(shape)
            }
        }
    }

    /// Chained Operations (Fused)
    pub fn sin(&self) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::Sin);
        new
    }
    pub fn cos(&self) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::Cos);
        new
    }
    pub fn exp(&self) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::Exp);
        new
    }
    pub fn sigmoid(&self) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::Sigmoid);
        new
    }
    pub fn add(&self, val: f64) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::AddScalar(val));
        new
    }
    pub fn mul(&self, val: f64) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::MulScalar(val));
        new
    }
    pub fn abs(&self) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::Abs);
        new
    }
    pub fn pow(&self, val: f64) -> Self {
        let mut new = self.clone();
        new.recipe.push(FusedOp::PowScalar(val));
        new
    }

    /// Execution
    pub fn load(&self) -> PyResult<Array> {
        let base_arr = match &self.source {
            LazySource::Memory(arr) => arr.clone(),
            LazySource::File { path, format } => match format {
                LazyFormat::RMath => rmath_load(path)?,
                LazyFormat::Csv   => csv_load(path)?,
                LazyFormat::Bin   => {
                    let s = self.shape.clone().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("No shape"))?;
                    bin_load(path, s)?
                }
            }
        };

        if self.recipe.is_empty() {
            return Ok(base_arr);
        }

        // Apply fusion
        let data = base_arr.data();
        let n = data.len();
        let out_data: Vec<f64> = if n >= crate::array::core::PAR_THRESHOLD {
            data.par_iter().map(|&x| self.apply_recipe(x)).collect()
        } else {
            data.iter().map(|&x| self.apply_recipe(x)).collect()
        };
        Ok(Array::from_flat(out_data, base_arr.shape.clone()))
    }

    /// Execute the fused pipeline and return the result as an Array.
    ///
    /// Example:
    ///     >>> lazy = ra.ones(10, 10).lazy().mul(5).abs()
    ///     >>> arr = lazy.execute()
    pub fn execute(&self) -> PyResult<Array> { self.load() }

    pub fn load_rows(&self, start: usize, end: usize) -> PyResult<Array> {
        match &self.source {
            LazySource::File { path, format } => {
                let mut arr = match format {
                    LazyFormat::Csv   => csv_load_rows(path, start, end)?,
                    LazyFormat::RMath => rmath_load_rows(path, start, end)?,
                    LazyFormat::Bin   => {
                        let shape = self.shape.clone().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("No shape"))?;
                        bin_load_rows(path, &shape, start, end)?
                    }
                };
                if !self.recipe.is_empty() {
                    let data = arr.data();
                    let out: Vec<f64> = data.iter().map(|&x| self.apply_recipe(x)).collect();
                    arr = Array::from_flat(out, arr.shape.clone());
                }
                Ok(arr)
            }
            LazySource::Memory(arr) => {
                let r = arr.nrows();
                let end = end.min(r);
                if start >= end { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Bounds error")); }
                
                let contig = arr.to_contiguous();
                let d = contig.data();
                let c = arr.ncols();
                let mut out_data = d[start*c..end*c].to_vec();
                if !self.recipe.is_empty() {
                    for x in out_data.iter_mut() { *x = self.apply_recipe(*x); }
                }
                Ok(Array::from_flat(out_data, vec![end-start, c]))
            }
        }
    }

    pub fn chunks<'py>(&self, py: Python<'py>, chunk_size: usize) -> PyResult<Bound<'py, PyAny>> {
        let path = match &self.source {
            LazySource::File { path, .. } => path.clone(),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Chunking only for files")),
        };
        let fmt = match &self.source {
            LazySource::File { format, .. } => format.clone(),
            _ => unreachable!(),
        };
        let iterator = ChunkIterator {
            path, format: fmt, shape: self.shape.clone(),
            chunk_size, current: 0, total_rows: None,
            recipe: self.recipe.clone(),
        };
        Ok(iterator.into_pyobject(py)?.into_any())
    }

    pub fn sum(&self) -> PyResult<f64> {
        let base_arr = match &self.source {
            LazySource::Memory(arr) => arr.clone(),
            LazySource::File { .. } => self.load()?, // For files, we still load for now
        };

        let data = base_arr.data();
        let n = data.len();
        
        let total: f64 = if n >= crate::array::core::PAR_THRESHOLD {
            data.par_iter().map(|&x| self.apply_recipe(x)).sum()
        } else {
            data.iter().map(|&x| self.apply_recipe(x)).sum()
        };
        Ok(total)
    }

    pub fn __repr__(&self) -> String {
        format!("LazyArray(source={:?}, steps={})", self.source, self.recipe.len())
    }
}

// ─── Iterators ───────────────────────────────────────────────────────────

/// An iterator over chunks of a large dataset.
#[pyclass(module = "rmath.array")]
pub struct ChunkIterator {
    path: String, format: LazyFormat, shape: Option<Vec<usize>>,
    chunk_size: usize, current: usize, total_rows: Option<usize>,
    recipe: Vec<FusedOp>,
}

#[pymethods]
impl ChunkIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self) -> PyResult<Option<Array>> {
        if self.total_rows.is_none() {
            self.total_rows = Some(match self.format {
                LazyFormat::Csv => csv_count_shape(&self.path)?.0,
                LazyFormat::RMath => rmath_read_header(&self.path)?[0],
                LazyFormat::Bin => self.shape.as_ref().unwrap()[0],
            });
        }
        let total = self.total_rows.unwrap();
        if self.current >= total { return Ok(None); }
        let end = (self.current + self.chunk_size).min(total);
        let start = self.current; self.current = end;
        
        let mut chunk = match self.format {
            LazyFormat::Csv => csv_load_rows(&self.path, start, end)?,
            LazyFormat::RMath => rmath_load_rows(&self.path, start, end)?,
            LazyFormat::Bin => bin_load_rows(&self.path, self.shape.as_ref().unwrap(), start, end)?,
        };
        if !self.recipe.is_empty() {
            let out: Vec<f64> = chunk.data().iter().map(|&x| {
                let mut val = x;
                for op in &self.recipe {
                    match op {
                        FusedOp::Sin => val = val.sin(),
                        FusedOp::Cos => val = val.cos(),
                        FusedOp::Exp => val = val.exp(),
                        FusedOp::Tanh => val = val.tanh(),
                        FusedOp::Abs => val = val.abs(),
                        FusedOp::Sigmoid => val = 1.0 / (1.0 + (-val).exp()),
                        FusedOp::AddScalar(s) => val += s,
                        FusedOp::MulScalar(s) => val *= s,
                        FusedOp::PowScalar(s) => val = val.powf(*s),
                    }
                }
                val
            }).collect();
            chunk = Array::from_flat(out, chunk.shape.clone());
        }
        Ok(Some(chunk))
    }
}

// Rest of IO... (I will use the original implementation for these to be safe)

/// Memory-mapped array for large out-of-core datasets.
#[pyclass(module = "rmath.array")]
pub struct MmapArray {
    pub path: String, pub rows: usize, pub cols: usize,
    mmap: Option<memmap2::Mmap>,
}

#[pymethods]
impl MmapArray {
    #[staticmethod]
    pub fn mmap(path: String, rows: usize, cols: usize) -> PyResult<Self> {
        let file = File::open(&path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(MmapArray { path, rows, cols, mmap: Some(mmap) })
    }
    #[getter] pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }
    pub fn load_all(&self) -> PyResult<Array> {
        let ptr = self.mmap.as_ref().unwrap().as_ptr() as *const f64;
        let n = self.rows * self.cols;
        let data = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
        Ok(Array::from_flat(data, vec![self.rows, self.cols]))
    }
    pub fn get_row(&self, i: usize) -> PyResult<Vec<f64>> {
        let ptr = self.mmap.as_ref().unwrap().as_ptr() as *const f64;
        let row = unsafe { std::slice::from_raw_parts(ptr.add(i*self.cols), self.cols) }.to_vec();
        Ok(row)
    }
    pub fn get_element(&self, r: usize, c: usize) -> PyResult<f64> {
        let ptr = self.mmap.as_ref().unwrap().as_ptr() as *const f64;
        Ok(unsafe { *ptr.add(r*self.cols + c) })
    }
    pub fn chunks<'py>(&self, py: Python<'py>, chunk_size: usize) -> PyResult<Bound<'py, PyAny>> {
        let iter = MmapChunkIterator { path: self.path.clone(), rows: self.rows, cols: self.cols, chunk_size, current: 0 };
        Ok(iter.into_pyobject(py)?.into_any())
    }
    pub fn load_rows(&self, start: usize, end: usize) -> PyResult<Array> {
        let r = self.rows;
        let c = self.cols;
        let end = end.min(r);
        if start >= end { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Bounds error")); }
        let ptr = self.mmap.as_ref().unwrap().as_ptr() as *const f64;
        let data = unsafe { std::slice::from_raw_parts(ptr.add(start*c), (end-start)*c) }.to_vec();
        Ok(Array::from_flat(data, vec![end-start, c]))
    }
}

#[pyclass(module = "rmath")]
pub struct MmapChunkIterator {
    path: String, rows: usize, cols: usize, chunk_size: usize, current: usize,
}
#[pymethods]
impl MmapChunkIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self) -> PyResult<Option<Array>> {
        if self.current >= self.rows { return Ok(None); }
        let end = (self.current + self.chunk_size).min(self.rows);
        let f = File::open(&self.path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&f) }.unwrap();
        let ptr = mmap.as_ptr() as *const f64;
        let n = (end - self.current) * self.cols;
        let data = unsafe { std::slice::from_raw_parts(ptr.add(self.current*self.cols), n) }.to_vec();
        let chunk = Array::from_flat(data, vec![end - self.current, self.cols]);
        self.current = end;
        Ok(Some(chunk))
    }
}

// IO HELPERS (Complete)
fn csv_count_shape(path: &str) -> PyResult<(usize, usize)> {
    let f = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let r = BufReader::new(f);
    let mut rows = 0; let mut cols = 0;
    for l in r.lines() {
        let l = l.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        if l.trim().is_empty() { continue; }
        if rows == 0 { cols = l.split(',').count(); }
        rows += 1;
    }
    Ok((rows, cols))
}
fn csv_load_rows(path: &str, start: usize, end: usize) -> PyResult<Array> {
    let f = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let r = BufReader::new(f);
    let mut data = Vec::new(); let mut cols = 0; let mut idx = 0;
    for l in r.lines() {
        let l = l.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        if l.trim().is_empty() { continue; }
        if idx >= start && idx < end {
            let row: Vec<f64> = l.split(',').map(|s| s.trim().parse().unwrap_or(0.0)).collect();
            if cols == 0 { cols = row.len(); }
            data.extend(row);
        }
        idx += 1;
    }
    let n = data.len();
    let rows = if cols > 0 { n / cols } else { 0 };
    Ok(Array::from_flat(data, vec![rows, cols]))
}
pub fn rmath_read_header(path: &str) -> PyResult<Vec<usize>> {
    use std::io::Read;
    let mut f = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let mut h = [0u8; 7]; f.read_exact(&mut h).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let ndim = u16::from_le_bytes([h[5], h[6]]) as usize;
    let mut s_buf = vec![0u8; ndim * 8]; f.read_exact(&mut s_buf).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok((0..ndim).map(|i| u64::from_le_bytes(s_buf[i*8..(i+1)*8].try_into().unwrap()) as usize).collect())
}
fn rmath_load_rows(path: &str, start: usize, end: usize) -> PyResult<Array> {
    use std::io::{Read, Seek, SeekFrom};
    let s = rmath_read_header(path)?; let c = s[1];
    let offset = 7 + s.len()*8 + start*c*8;
    let mut f = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    f.seek(SeekFrom::Start(offset as u64)).unwrap();
    let n = (end - start)*c; let mut buf = vec![0u8; n*8]; f.read_exact(&mut buf).unwrap();
    let data = (0..n).map(|i| f64::from_le_bytes(buf[i*8..(i+1)*8].try_into().unwrap())).collect();
    Ok(Array::from_flat(data, vec![end-start, c]))
}
fn csv_load(path: &str) -> PyResult<Array> {
    let (r, _c) = csv_count_shape(path)?;
    csv_load_rows(path, 0, r)
}
fn rmath_load(path: &str) -> PyResult<Array> {
    let s = rmath_read_header(path)?;
    rmath_load_rows(path, 0, s[0])
}
fn bin_load(path: &str, s: Vec<usize>) -> PyResult<Array> {
    bin_load_rows(path, &s, 0, s[0])
}
fn bin_load_rows(path: &str, s: &[usize], start: usize, end: usize) -> PyResult<Array> {
    use std::io::{Read, Seek, SeekFrom};
    let c = s[1]; let offset = start*c*8;
    let mut f = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    f.seek(SeekFrom::Start(offset as u64)).unwrap();
    let n = (end - start)*c; let mut buf = vec![0u8; n*8]; f.read_exact(&mut buf).unwrap();
    let data = (0..n).map(|i| f64::from_le_bytes(buf[i*8..(i+1)*8].try_into().unwrap())).collect();
    Ok(Array::from_flat(data, vec![end-start, c]))
}
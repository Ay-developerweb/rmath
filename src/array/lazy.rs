use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use super::core::Array;

// ─── LazyArray ───────────────────────────────────────────────────────────────
// 
// LazyArray wraps a file path + shape metadata and reads chunks on demand.
// Actual data is NOT loaded until .load() or iteration is called.
// Supports: .rmath binary, CSV, memory-mapped f64 binary (.bin)

#[pyclass(module = "rmath")]
pub struct LazyArray {
    pub path:   String,
    pub format: LazyFormat,
    pub shape:  Option<Vec<usize>>,  // known after header read
    pub dtype:  String,              // "f64" default
}

#[derive(Clone, Debug)]
pub enum LazyFormat {
    RMath,   // custom binary .rmath
    Csv,     // delimited text
    Bin,     // raw f64 little-endian binary
}

#[pymethods]
impl LazyArray {

    /// Open a lazy reference to a file — does NOT read data
    #[staticmethod]
    pub fn open(path: String) -> PyResult<Self> {
        let fmt = if path.ends_with(".rmath")     { LazyFormat::RMath }
                  else if path.ends_with(".csv")  { LazyFormat::Csv }
                  else if path.ends_with(".bin")  { LazyFormat::Bin }
                  else {
                      return Err(pyo3::exceptions::PyValueError::new_err(
                          "Unsupported format. Use .rmath / .csv / .bin"));
                  };
        Ok(LazyArray { path, format: fmt, shape: None, dtype: "f64".into() })
    }

    /// Peek at shape/metadata WITHOUT loading data
    pub fn peek(&mut self) -> PyResult<Vec<usize>> {
        match self.format {
            LazyFormat::RMath => {
                let shape = rmath_read_header(&self.path)?;
                self.shape = Some(shape.clone());
                Ok(shape)
            }
            LazyFormat::Csv => {
                let (r, c) = csv_count_shape(&self.path)?;
                let shape = vec![r, c];
                self.shape = Some(shape.clone());
                Ok(shape)
            }
            LazyFormat::Bin => {
                Err(pyo3::exceptions::PyValueError::new_err(
                    ".bin files need explicit shape — use LazyArray.open_bin(path, rows, cols)"))
            }
        }
    }

    /// Fully load into Array
    pub fn load(&self) -> PyResult<Array> {
        match self.format {
            LazyFormat::RMath => rmath_load(&self.path),
            LazyFormat::Csv   => csv_load(&self.path),
            LazyFormat::Bin   => {
                let shape = self.shape.clone().ok_or_else(||
                    pyo3::exceptions::PyValueError::new_err("Call peek() or open_bin() first"))?;
                bin_load(&self.path, shape)
            }
        }
    }

    /// Load a slice of rows [start_row, end_row) — low-memory
    pub fn load_rows(&self, start: usize, end: usize) -> PyResult<Array> {
        match self.format {
            LazyFormat::Csv   => csv_load_rows(&self.path, start, end),
            LazyFormat::RMath => rmath_load_rows(&self.path, start, end),
            LazyFormat::Bin   => {
                let shape = self.shape.clone().ok_or_else(||
                    pyo3::exceptions::PyValueError::new_err("Call open_bin(path, rows, cols) first"))?;
                bin_load_rows(&self.path, &shape, start, end)
            }
        }
    }

    /// Iterate over chunks of `chunk_size` rows — yields Array objects
    /// Python usage: for batch in lazy.chunks(1000): ...
    pub fn chunks<'py>(&self, py: Python<'py>, chunk_size: usize) -> PyResult<Bound<'py, PyAny>> {
        let iterator = ChunkIterator {
            path:       self.path.clone(),
            format:     self.format.clone(),
            shape:      self.shape.clone(),
            chunk_size,
            current:    0,
            total_rows: None,
        };
        Ok(iterator.into_pyobject(py)?.into_any())
    }

    /// Memory-map a .bin file for zero-copy access (read-only view)
    #[staticmethod]
    pub fn mmap(path: String, rows: usize, cols: usize) -> PyResult<MmapArray> {
        MmapArray::new(path, rows, cols)
    }

    /// Convenience: open a raw binary file with known shape
    #[staticmethod]
    pub fn open_bin(path: String, rows: usize, cols: usize) -> Self {
        LazyArray {
            path,
            format: LazyFormat::Bin,
            shape: Some(vec![rows, cols]),
            dtype: "f64".into(),
        }
    }

    pub fn __repr__(&self) -> String {
        format!("LazyArray(path={:?}, format={:?}, shape={:?})", self.path, self.format, self.shape)
    }

    pub fn close(&self) {
        // No-op for standard LazyArray since files are opened per operation.
        // Provided for context manager compatibility.
    }

    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    #[pyo3(signature = (_exc_type, _exc_value, _traceback))]
    pub fn __exit__(&self, _exc_type: &Bound<'_, PyAny>, _exc_value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) {
        self.close();
    }
}

// ─── ChunkIterator ───────────────────────────────────────────────────────────

#[pyclass(module = "rmath")]
pub struct ChunkIterator {
    path:       String,
    format:     LazyFormat,
    shape:      Option<Vec<usize>>,
    chunk_size: usize,
    current:    usize,
    total_rows: Option<usize>,
}

#[pymethods]
impl ChunkIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self) -> PyResult<Option<Array>> {
        // Determine total rows lazily
        if self.total_rows.is_none() {
            let total = match self.format {
                LazyFormat::Csv => {
                    let (r, _) = csv_count_shape(&self.path)?;
                    r
                }
                LazyFormat::RMath => {
                    let shape = rmath_read_header(&self.path)?;
                    shape[0]
                }
                LazyFormat::Bin => {
                    self.shape.as_ref().map(|s| s[0]).unwrap_or(0)
                }
            };
            self.total_rows = Some(total);
        }

        let total = self.total_rows.unwrap();
        if self.current >= total { return Ok(None); }

        let end = (self.current + self.chunk_size).min(total);
        let start = self.current;
        self.current = end;

        let chunk = match self.format {
            LazyFormat::Csv   => csv_load_rows(&self.path, start, end)?,
            LazyFormat::RMath => rmath_load_rows(&self.path, start, end)?,
            LazyFormat::Bin   => {
                let shape = self.shape.clone().unwrap();
                bin_load_rows(&self.path, &shape, start, end)?
            }
        };
        Ok(Some(chunk))
    }
}

// ─── MmapArray ───────────────────────────────────────────────────────────────
// Zero-copy memory-mapped f64 array. Data stays on disk until accessed.

#[pyclass(module = "rmath")]
pub struct MmapArray {
    pub path:  String,
    pub rows:  usize,
    pub cols:  usize,
    mmap:      Option<memmap2::Mmap>,
}

impl MmapArray {
    pub fn new(path: String, rows: usize, cols: usize) -> PyResult<Self> {
        let file = File::open(&path).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("mmap failed: {}", e)))?;
        Ok(MmapArray { path, rows, cols, mmap: Some(mmap) })
    }

    fn as_f64_slice(&self) -> PyResult<&[f64]> {
        if let Some(mmap) = &self.mmap {
            let ptr = mmap.as_ptr() as *const f64;
            let len = mmap.len() / 8;
            Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("MmapArray is closed"))
        }
    }
}

#[pymethods]
impl MmapArray {
    #[staticmethod]
    pub fn mmap(path: String, rows: usize, cols: usize) -> PyResult<Self> {
        Self::new(path, rows, cols)
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }

    pub fn get_row(&self, i: usize) -> PyResult<Vec<f64>> {
        if i >= self.rows { return Err(pyo3::exceptions::PyIndexError::new_err("Row out of bounds")); }
        let s = self.as_f64_slice()?;
        Ok(s[i*self.cols..(i+1)*self.cols].to_vec())
    }

    pub fn get_element(&self, row: usize, col: usize) -> PyResult<f64> {
        if row >= self.rows || col >= self.cols {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of bounds"));
        }
        Ok(self.as_f64_slice()?[row * self.cols + col])
    }

    pub fn load_rows(&self, start: usize, end: usize) -> PyResult<Array> {
        let end = end.min(self.rows);
        if start >= end { return Err(pyo3::exceptions::PyValueError::new_err("start >= end")); }
        let s = self.as_f64_slice()?;
        let data = s[start*self.cols..end*self.cols].to_vec();
        Ok(Array::from_flat(data, vec![end-start, self.cols]))
    }

    pub fn load_all(&self) -> PyResult<Array> {
        let data = self.as_f64_slice()?.to_vec();
        Ok(Array::from_flat(data, vec![self.rows, self.cols]))
    }

    pub fn close(&mut self) {
        self.mmap = None;
    }

    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    #[pyo3(signature = (_exc_type, _exc_value, _traceback))]
    pub fn __exit__(&mut self, _exc_type: &Bound<'_, PyAny>, _exc_value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) {
        self.close();
    }

    /// Memory-mapped chunk iterator — yields Array chunks without copying until needed
    pub fn chunks<'py>(&self, py: Python<'py>, chunk_size: usize) -> PyResult<Bound<'py, PyAny>> {
        let iter = MmapChunkIterator {
            path: self.path.clone(),
            rows: self.rows,
            cols: self.cols,
            chunk_size,
            current: 0,
        };
        Ok(iter.into_pyobject(py)?.into_any())
    }

    pub fn __repr__(&self) -> String {
        format!("MmapArray(path={:?}, shape=({}, {}))", self.path, self.rows, self.cols)
    }
}

#[pyclass(module = "rmath")]
pub struct MmapChunkIterator {
    path:       String,
    rows:       usize,
    cols:       usize,
    chunk_size: usize,
    current:    usize,
}

#[pymethods]
impl MmapChunkIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&mut self) -> PyResult<Option<Array>> {
        if self.current >= self.rows { return Ok(None); }
        let end = (self.current + self.chunk_size).min(self.rows);
        let arr = MmapArray::new(self.path.clone(), self.rows, self.cols)?;
        let chunk = arr.load_rows(self.current, end)?;
        self.current = end;
        Ok(Some(chunk))
    }
}

// ─── Format implementations ──────────────────────────────────────────────────

fn csv_count_shape(path: &str) -> PyResult<(usize, usize)> {
    let file = File::open(path).map_err(|e|
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;
    let reader = BufReader::new(file);
    let mut rows = 0usize;
    let mut cols = 0usize;
    for line in reader.lines() {
        let line = line.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let line = line.trim().to_string();
        if line.is_empty() { continue; }
        if rows == 0 { cols = line.split(',').count(); }
        rows += 1;
    }
    Ok((rows, cols))
}

fn csv_load(path: &str) -> PyResult<Array> {
    let (r, c) = csv_count_shape(path)?;
    csv_load_rows(path, 0, r).map(|mut a| {
        a.shape = vec![r, c];
        a
    })
}

fn csv_load_rows(path: &str, start: usize, end: usize) -> PyResult<Array> {
    let file = File::open(path).map_err(|e|
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    let mut cols = 0usize;
    let mut row_idx = 0usize;

    for line in reader.lines() {
        let line = line.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let line = line.trim().to_string();
        if line.is_empty() { continue; }
        if row_idx >= end { break; }
        if row_idx >= start {
            let parsed: Vec<f64> = line.split(',').map(|s| {
                s.trim().parse::<f64>().unwrap_or(0.0)
            }).collect();
            if cols == 0 { cols = parsed.len(); }
            data.extend_from_slice(&parsed);
        }
        row_idx += 1;
    }

    let rows = if cols > 0 { data.len() / cols } else { 0 };
    Ok(Array::from_flat(data, vec![rows, cols]))
}

// ── .rmath binary format ──────────────────────────────────────────────────────
// Layout:
//   [0..4]   magic: b"RMTH"
//   [4]      version: u8 = 1
//   [5..6]   ndim: u16 le
//   [6..6+ndim*8] shape: ndim × u64 le
//   [rest]   data: f64 le flat row-major
const MAGIC: &[u8; 4] = b"RMTH";
const VERSION: u8 = 1;

pub fn rmath_read_header(path: &str) -> PyResult<Vec<usize>> {
    use std::io::Read;
    let mut f = File::open(path).map_err(|e|
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;
    let mut buf = [0u8; 7];
    f.read_exact(&mut buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    if &buf[0..4] != MAGIC {
        return Err(pyo3::exceptions::PyValueError::new_err("Not a .rmath file (bad magic)"));
    }
    if buf[4] != VERSION {
        return Err(pyo3::exceptions::PyValueError::new_err("Unsupported .rmath version"));
    }
    let ndim = u16::from_le_bytes([buf[5], buf[6]]) as usize;
    let mut shape_buf = vec![0u8; ndim * 8];
    f.read_exact(&mut shape_buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let shape: Vec<usize> = (0..ndim).map(|i| {
        u64::from_le_bytes(shape_buf[i*8..(i+1)*8].try_into().unwrap()) as usize
    }).collect();
    Ok(shape)
}

fn rmath_load(path: &str) -> PyResult<Array> {
    use std::io::Read;
    let shape = rmath_read_header(path)?;
    let header_len = 7 + shape.len() * 8;
    let n: usize = shape.iter().product();
    let mut f = File::open(path).map_err(|e|
        pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut discard = vec![0u8; header_len];
    f.read_exact(&mut discard).unwrap();
    let mut data_buf = vec![0u8; n * 8];
    f.read_exact(&mut data_buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let data: Vec<f64> = (0..n).map(|i| {
        f64::from_le_bytes(data_buf[i*8..(i+1)*8].try_into().unwrap())
    }).collect();
    Ok(Array::from_flat(data, shape))
}

fn rmath_load_rows(path: &str, start: usize, end: usize) -> PyResult<Array> {
    use std::io::{Read, Seek, SeekFrom};
    let shape = rmath_read_header(path)?;
    if shape.len() < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Not 2-D")); }
    let (total_r, c) = (shape[0], shape[1]);
    let end = end.min(total_r);
    if start >= end { return Err(pyo3::exceptions::PyValueError::new_err("start >= end")); }
    let header_len = 7 + shape.len() * 8;
    let offset = header_len + start * c * 8;
    let n = (end - start) * c;
    let mut f = File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    f.seek(SeekFrom::Start(offset as u64)).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut buf = vec![0u8; n * 8];
    f.read_exact(&mut buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let data: Vec<f64> = (0..n).map(|i| {
        f64::from_le_bytes(buf[i*8..(i+1)*8].try_into().unwrap())
    }).collect();
    Ok(Array::from_flat(data, vec![end - start, c]))
}

fn bin_load(path: &str, shape: Vec<usize>) -> PyResult<Array> {
    use std::io::Read;
    let n: usize = shape.iter().product();
    let mut f = File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut buf = vec![0u8; n * 8];
    f.read_exact(&mut buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let data: Vec<f64> = (0..n).map(|i| {
        f64::from_le_bytes(buf[i*8..(i+1)*8].try_into().unwrap())
    }).collect();
    Ok(Array::from_flat(data, shape))
}

fn bin_load_rows(path: &str, shape: &[usize], start: usize, end: usize) -> PyResult<Array> {
    use std::io::{Read, Seek, SeekFrom};
    if shape.len() < 2 { return Err(pyo3::exceptions::PyValueError::new_err("Need 2-D shape")); }
    let (_, c) = (shape[0], shape[1]);
    let end = end.min(shape[0]);
    let n = (end - start) * c;
    let offset = start * c * 8;
    let mut f = File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    f.seek(SeekFrom::Start(offset as u64)).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut buf = vec![0u8; n * 8];
    f.read_exact(&mut buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let data: Vec<f64> = (0..n).map(|i| {
        f64::from_le_bytes(buf[i*8..(i+1)*8].try_into().unwrap())
    }).collect();
    Ok(Array::from_flat(data, vec![end - start, c]))
}
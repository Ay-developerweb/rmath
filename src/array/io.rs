use pyo3::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write, BufReader, BufRead};
use super::core::Array;
use super::lazy::{rmath_read_header};

const MAGIC: &[u8; 4] = b"RMTH";
const VERSION: u8 = 1;

#[pymethods]
impl Array {

    // ── .rmath binary format ──────────────────────────────────────────────
    // Fastest: header + raw f64 LE bytes. No reprocessing on load.

    /// Save to .rmath binary format.
    ///
    /// This is the fastest persistence format for RMath, storing the raw 
    /// memory buffer with a minimal header.
    ///
    /// Example:
    ///     >>> a.save("data.rmath")
    pub fn save(&self, path: &str) -> PyResult<()> {
        let f = File::create(path).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("Cannot create {}: {}", path, e)))?;
        let mut w = BufWriter::new(f);

        // Header: magic(4) + version(1) + ndim(2) + shape(ndim*8)
        w.write_all(MAGIC).unwrap();
        w.write_all(&[VERSION]).unwrap();
        let ndim = self.shape.len() as u16;
        w.write_all(&ndim.to_le_bytes()).unwrap();
        for &s in &self.shape {
            w.write_all(&(s as u64).to_le_bytes()).unwrap();
        }

        // Data: flat f64 LE
        for &x in self.data() {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
        w.flush().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load from .rmath binary format — instant, no reprocessing
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        use std::io::Read;
        let shape = rmath_read_header(path)?;
        let header_len = 7 + shape.len() * 8;
        let n: usize = shape.iter().product();

        let mut f = File::open(path).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut discard = vec![0u8; header_len];
        f.read_exact(&mut discard).unwrap();

        let mut buf = vec![0u8; n * 8];
        f.read_exact(&mut buf).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("Read error: {}", e)))?;

        let data: Vec<f64> = (0..n).map(|i| {
            f64::from_le_bytes(buf[i*8..(i+1)*8].try_into().unwrap())
        }).collect();

        Ok(Self::from_flat(data, shape))
    }

    /// Save to raw binary (no header) — use with LazyArray.open_bin()
    pub fn save_bin(&self, path: &str) -> PyResult<()> {
        let f = File::create(path).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("Cannot create {}: {}", path, e)))?;
        let mut w = BufWriter::new(f);
        for &x in self.data() {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
        w.flush().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    // ── CSV ───────────────────────────────────────────────────────────────

    /// Save as CSV
    #[pyo3(signature = (path, header=None))]
    pub fn save_csv(&self, path: &str, header: Option<Vec<String>>) -> PyResult<()> {
        let f = File::create(path).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("Cannot create {}: {}", path, e)))?;
        let mut w = BufWriter::new(f);

        if let Some(h) = header {
            writeln!(w, "{}", h.join(",")).unwrap();
        }

        let (r, c) = (self.nrows(), self.ncols());
        let d = self.data();
        for i in 0..r {
            let row: Vec<String> = (0..c).map(|j| format!("{}", d[i*c+j])).collect();
            writeln!(w, "{}", row.join(",")).unwrap();
        }
        w.flush().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load from CSV (no header row assumed; use load_csv_with_header for headed files)
    #[staticmethod]
    pub fn load_csv(path: &str) -> PyResult<Self> {
        let f = File::open(path).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;
        let reader = BufReader::new(f);
        let mut data = Vec::new();
        let mut cols = 0usize;
        let mut rows = 0usize;
        for line in reader.lines() {
            let line = line.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let line = line.trim().to_string();
            if line.is_empty() { continue; }
            let parsed: Vec<f64> = line.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
                .collect();
            if cols == 0 { cols = parsed.len(); }
            data.extend_from_slice(&parsed);
            rows += 1;
        }
        Ok(Self::from_flat(data, vec![rows, cols]))
    }

    /// Load CSV, skip first row (header), return (Array, Vec<String> column names)
    #[staticmethod]
    pub fn load_csv_with_header(path: &str) -> PyResult<(Self, Vec<String>)> {
        let f = File::open(path).map_err(|e|
            pyo3::exceptions::PyIOError::new_err(format!("Cannot open {}: {}", path, e)))?;
        let reader = BufReader::new(f);
        let mut lines = reader.lines();
        let header_line = lines.next()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Empty file"))??;
        let header: Vec<String> = header_line.split(',').map(|s| s.trim().to_string()).collect();

        let mut data = Vec::new();
        let cols = header.len();
        let mut rows = 0usize;
        for line in lines {
            let line = line.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let line = line.trim().to_string();
            if line.is_empty() { continue; }
            let parsed: Vec<f64> = line.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
                .collect();
            data.extend_from_slice(&parsed);
            rows += 1;
        }
        Ok((Self::from_flat(data, vec![rows, cols]), header))
    }

    // ── Safetensors ───────────────────────────────────────────────────────
    // Industry standard zero-copy persistence.

    /// Save array to a safetensors file with a specific key
    pub fn save_safetensors(&self, path: &str, key: &str) -> PyResult<()> {
        let d = self.data();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * 8)
        };
        
        let mut tensors = std::collections::HashMap::new();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F64,
            self.shape.clone(),
            bytes
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        tensors.insert(key.to_string(), view);
        safetensors::tensor::serialize_to_file(tensors, &None, std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load a specific key from a safetensors file
    #[staticmethod]
    pub fn load_safetensors(path: &str, key: &str) -> PyResult<Self> {
        let buffer = std::fs::read(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let st = safetensors::SafeTensors::deserialize(&buffer)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        let view = st.tensor(key).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let shape = view.shape().to_vec();
        let data_bytes = view.data();
        
        let n = data_bytes.len() / 8;
        let data: Vec<f64> = (0..n).map(|i| {
            f64::from_le_bytes(data_bytes[i*8..(i+1)*8].try_into().unwrap())
        }).collect();
        
        Ok(Self::from_flat(data, shape))
    }

    // ── Info ──────────────────────────────────────────────────────────────

    /// File size in bytes that save() would write
    pub fn estimated_bytes(&self) -> usize {
        7 + self.shape.len() * 8 + self.len() * 8
    }

    pub fn memory_bytes(&self) -> usize {
        self.len() * 8
    }
}
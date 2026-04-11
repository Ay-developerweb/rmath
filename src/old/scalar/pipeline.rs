use pyo3::prelude::*;
use crate::vector::Vector;
use rayon::prelude::*;
use multiversion::multiversion;
use std::sync::Arc;

const CHUNK_SIZE: usize = 64;

#[derive(Clone, Copy)]
enum Op {
    Sin,
    Cos,
    Sqrt,
    Abs,
    Exp,
    Add(f64),
    Mul(f64),
}

#[derive(Clone)]
enum PipelineStep {
    Map(Op),
    FilterGreater(f64),
    FilterLess(f64),
}

#[derive(Clone)]
enum Source {
    Range { start: f64, stop: f64, step: f64 },
    Buffer(Arc<Vec<f64>>),
}

#[pyclass]
#[derive(Clone)]
pub struct LazyPipeline {
    source: Source,
    steps: Vec<PipelineStep>,
}

#[pymethods]
impl LazyPipeline {
    pub fn to_vector(&self) -> Vector {
        let n = self.len();
        let data: Vec<f64> = if n < 10000 {
            self.collect_serial(0, n)
        } else {
            self.collect_parallel(n)
        };
        Vector::new(data)
    }

    pub fn sum(&self) -> f64 {
        let n = self.len();
        if n == 0 { return 0.0; }

        if n < 10000 {
            self.sum_strips(0, n)
        } else {
            let n_threads = rayon::current_num_threads();
            let chunk_per_thread = (n + n_threads - 1) / n_threads;

            (0..n_threads)
                .into_par_iter()
                .map(|i| {
                    let start_idx = i * chunk_per_thread;
                    let end_idx = (start_idx + chunk_per_thread).min(n);
                    if start_idx < end_idx {
                        self.sum_strips(start_idx, end_idx)
                    } else {
                        0.0
                    }
                })
                .sum()
        }
    }

    pub fn mean(&self) -> f64 {
        let n = self.len();
        if n == 0 { return f64::NAN; }
        let has_filters = self.steps.iter().any(|s| !matches!(s, PipelineStep::Map(_)));
        if has_filters {
             let v = self.to_vector();
             v.mean().unwrap_or(f64::NAN)
        } else {
            self.sum() / n as f64
        }
    }

    pub fn max(&self) -> f64 {
        let n = self.len();
        if n == 0 { return f64::NAN; }
        let n_threads = rayon::current_num_threads();
        let chunk_per_thread = (n + n_threads - 1) / n_threads;

        (0..n_threads)
            .into_par_iter()
            .map(|i| {
                let start_idx = i * chunk_per_thread;
                let end_idx = (start_idx + chunk_per_thread).min(n);
                if start_idx < end_idx {
                    self.max_strips(start_idx, end_idx)
                } else {
                    f64::NEG_INFINITY
                }
            })
            .reduce(|| f64::NEG_INFINITY, |a, b| a.max(b))
    }

    pub fn min(&self) -> f64 {
        let n = self.len();
        if n == 0 { return f64::NAN; }
        let n_threads = rayon::current_num_threads();
        let chunk_per_thread = (n + n_threads - 1) / n_threads;

        (0..n_threads)
            .into_par_iter()
            .map(|i| {
                let start_idx = i * chunk_per_thread;
                let end_idx = (start_idx + chunk_per_thread).min(n);
                if start_idx < end_idx {
                    self.min_strips(start_idx, end_idx)
                } else {
                    f64::INFINITY
                }
            })
            .reduce(|| f64::INFINITY, |a, b| a.min(b))
    }

    // --- Chaining Methods ---
    pub fn sin(&mut self) -> Self { self.steps.push(PipelineStep::Map(Op::Sin)); self.clone() }
    pub fn cos(&mut self) -> Self { self.steps.push(PipelineStep::Map(Op::Cos)); self.clone() }
    pub fn sqrt(&mut self) -> Self { self.steps.push(PipelineStep::Map(Op::Sqrt)); self.clone() }
    pub fn abs(&mut self) -> Self { self.steps.push(PipelineStep::Map(Op::Abs)); self.clone() }
    pub fn exp(&mut self) -> Self { self.steps.push(PipelineStep::Map(Op::Exp)); self.clone() }
    pub fn add(&mut self, val: f64) -> Self { self.steps.push(PipelineStep::Map(Op::Add(val))); self.clone() }
    pub fn mul(&mut self, val: f64) -> Self { self.steps.push(PipelineStep::Map(Op::Mul(val))); self.clone() }
    pub fn filter_gt(&mut self, val: f64) -> Self { self.steps.push(PipelineStep::FilterGreater(val)); self.clone() }
    pub fn filter_lt(&mut self, val: f64) -> Self { self.steps.push(PipelineStep::FilterLess(val)); self.clone() }

    pub fn as_(&self, _target: &Bound<'_, PyAny>) -> Self { self.clone() }

    pub fn __repr__(&self) -> String {
        let src_type = match &self.source {
            Source::Range { .. } => "Range",
            Source::Buffer(_) => "Vector",
        };
        format!(
            "LazyPipeline(source={}, steps={})",
            src_type, self.steps.len()
        )
    }
}

impl LazyPipeline {
    fn len(&self) -> usize {
        match &self.source {
            Source::Range { start, stop, step } => (((stop - start) / step).ceil().max(0.0)) as usize,
            Source::Buffer(arc_vec) => arc_vec.len(),
        }
    }

    #[inline(always)]
    fn get_source_val(&self, idx: usize) -> f64 {
        match &self.source {
            Source::Range { start, step, .. } => start + idx as f64 * step,
            Source::Buffer(arc_vec) => arc_vec[idx],
        }
    }

    #[multiversion(targets("x86_64+avx2"))]
    #[inline(always)]
    fn apply_op_simd(op: &Op, chunk: &mut [f64]) {
        for x in chunk.iter_mut() {
            *x = match op {
                Op::Sin => x.sin(),
                Op::Cos => x.cos(),
                Op::Sqrt => x.sqrt(),
                Op::Abs => x.abs(),
                Op::Exp => x.exp(),
                Op::Add(v) => *x + v,
                Op::Mul(v) => *x * v,
            };
        }
    }

    fn apply_steps(&self, val: f64, buffer: &mut Vec<f64>) {
        let mut x = val;
        for step in &self.steps {
            match step {
                PipelineStep::Map(op) => {
                    x = match op {
                        Op::Sin => x.sin(), Op::Cos => x.cos(), Op::Sqrt => x.sqrt(),
                        Op::Abs => x.abs(), Op::Exp => x.exp(), Op::Add(v) => x + v, Op::Mul(v) => x * v,
                    };
                }
                PipelineStep::FilterGreater(v) => { if x <= *v { return; } }
                PipelineStep::FilterLess(v) => { if x >= *v { return; } }
            }
        }
        buffer.push(x);
    }

    fn collect_serial(&self, start_idx: usize, end_idx: usize) -> Vec<f64> {
        let mut res = Vec::new();
        for i in start_idx..end_idx {
            let val = self.get_source_val(i);
            self.apply_steps(val, &mut res);
        }
        res
    }

    fn collect_parallel(&self, total_n: usize) -> Vec<f64> {
        let n_threads = rayon::current_num_threads();
        let chunk_size = (total_n + n_threads - 1) / n_threads;

        (0..n_threads)
            .into_par_iter()
            .map(|i| {
                let start = i * chunk_size;
                let end = (start + chunk_size).min(total_n);
                if start < end {
                    self.collect_serial(start, end)
                } else {
                    Vec::new()
                }
            })
            .flatten()
            .collect()
    }

    fn sum_strips(&self, start_idx: usize, end_idx: usize) -> f64 {
        let n = end_idx - start_idx;
        let mut total = 0.0;
        let mut i = 0;
        let all_maps = self.steps.iter().all(|s| matches!(s, PipelineStep::Map(_)));

        if all_maps {
            let ops: Vec<Op> = self.steps.iter().filter_map(|s| {
                if let PipelineStep::Map(op) = s { Some(*op) } else { None }
            }).collect();

            while i + CHUNK_SIZE <= n {
                let mut buffer = [0.0; CHUNK_SIZE];
                let base_idx = start_idx + i;
                match &self.source {
                    Source::Range { start, step, .. } => {
                        for j in 0..CHUNK_SIZE { buffer[j] = start + (base_idx + j) as f64 * step; }
                    }
                    Source::Buffer(arc_vec) => {
                        buffer.copy_from_slice(&arc_vec[base_idx..base_idx + CHUNK_SIZE]);
                    }
                }

                for op in &ops { Self::apply_op_simd(op, &mut buffer); }
                for val in buffer { total += val; }
                i += CHUNK_SIZE;
            }
        }

        while i < n {
            let mut x = self.get_source_val(start_idx + i);
            let mut keep = true;
            for step in &self.steps {
                match step {
                    PipelineStep::Map(op) => {
                        x = match op {
                            Op::Sin => x.sin(), Op::Cos => x.cos(), Op::Sqrt => x.sqrt(),
                            Op::Abs => x.abs(), Op::Exp => x.exp(), Op::Add(v) => x + v, Op::Mul(v) => x * v,
                        };
                    }
                    PipelineStep::FilterGreater(v) => { if x <= *v { keep = false; break; } }
                    PipelineStep::FilterLess(v) => { if x >= *v { keep = false; break; } }
                }
            }
            if keep { total += x; }
            i += 1;
        }
        total
    }

    fn max_strips(&self, start_idx: usize, end_idx: usize) -> f64 {
        let n = end_idx - start_idx;
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..n {
            let mut val = self.get_source_val(start_idx + i);
            let mut keep = true;
            for step in &self.steps {
                match step {
                    PipelineStep::Map(op) => {
                        val = match op {
                            Op::Sin => val.sin(), Op::Cos => val.cos(), Op::Sqrt => val.sqrt(),
                            Op::Abs => val.abs(), Op::Exp => val.exp(), Op::Add(v) => val + v, Op::Mul(v) => val * v,
                        };
                    }
                    PipelineStep::FilterGreater(v) => { if val <= *v { keep = false; break; } }
                    PipelineStep::FilterLess(v) => { if val >= *v { keep = false; break; } }
                }
            }
            if keep { max_val = max_val.max(val); }
        }
        max_val
    }

    fn min_strips(&self, start_idx: usize, end_idx: usize) -> f64 {
        let n = end_idx - start_idx;
        let mut min_val = f64::INFINITY;
        for i in 0..n {
            let mut val = self.get_source_val(start_idx + i);
            let mut keep = true;
            for step in &self.steps {
                match step {
                    PipelineStep::Map(op) => {
                        val = match op {
                            Op::Sin => val.sin(), Op::Cos => val.cos(), Op::Sqrt => val.sqrt(),
                            Op::Abs => val.abs(), Op::Exp => val.exp(), Op::Add(v) => val + v, Op::Mul(v) => val * v,
                        };
                    }
                    PipelineStep::FilterGreater(v) => { if val <= *v { keep = false; break; } }
                    PipelineStep::FilterLess(v) => { if val >= *v { keep = false; break; } }
                }
            }
            if keep { min_val = min_val.min(val); }
        }
        min_val
    }
}

/// Creates a pipeline backed by a shared buffer (Instant/Zero-copy).
pub fn from_shared_buffer(buffer: Arc<Vec<f64>>) -> LazyPipeline {
    LazyPipeline {
        source: Source::Buffer(buffer),
        steps: Vec::new(),
    }
}

#[pyfunction]
#[pyo3(signature = (start, stop=None, step=1.0))]
pub fn loop_range(start: f64, stop: Option<f64>, step: f64) -> PyResult<LazyPipeline> {
    let (actual_start, actual_stop) = match stop { Some(s) => (start, s), None => (0.0, start) };
    if step == 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("step cannot be 0")); }
    Ok(LazyPipeline {
        source: Source::Range { start: actual_start, stop: actual_stop, step },
        steps: Vec::new(),
    })
}

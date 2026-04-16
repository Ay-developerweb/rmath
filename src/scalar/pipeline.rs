use pyo3::prelude::*;
use crate::vector::Vector;
use rayon::prelude::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Threshold below which we use serial execution.
// Rayon thread-spawn overhead dominates for small N — measured crossover is
// typically 8_000–20_000 elements depending on operation complexity.
// ---------------------------------------------------------------------------
const PARALLEL_THRESHOLD: usize = 10_000;

// ---------------------------------------------------------------------------
// Chunk size for the SIMD-friendly strip accumulation in sum/min/max.
// 64 f64 values = 512 bytes = fits in L1 cache line budget on most CPUs.
// ---------------------------------------------------------------------------
const CHUNK_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// Op — a single elementwise operation, Copy so it's free to clone.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
enum Op {
    Sin,
    Cos,
    Sqrt,
    Abs,
    Exp,
    Add(f64),
    Mul(f64),
    Sub(f64),  // x - val
    Div(f64),  // x / val (val != 0 enforced at construction)
}

// ---------------------------------------------------------------------------
// PipelineStep — either a map (preserves count) or a filter (may reduce count)
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
enum PipelineStep {
    Map(Op),
    FilterGreater(f64),   // keep x if x > val
    FilterLess(f64),      // keep x if x < val
    FilterFinite,         // keep x if x.is_finite()
}

// ---------------------------------------------------------------------------
// Source — what the pipeline draws values from
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
enum Source {
    /// A numeric range: start, stop, step (all f64 for fractional step support)
    Range { start: f64, stop: f64, step: f64 },
    /// A shared immutable buffer (zero-copy from Vector or from_list)
    Buffer(Arc<Vec<f64>>),
}

// ---------------------------------------------------------------------------
// LazyPipeline
//
// Builder pattern: every chainable method takes `self` by value, appends a
// step, and returns `self`.  This means:
//   - No mutation of the original (no surprising aliasing bugs)
//   - No clone per step (the old code cloned after mutating &mut self)
//   - Chains like p.sin().add(1.0).sqrt() compile to a single Vec::push loop
//
// The only PyO3 wrinkle: #[pymethods] requires `&self` or `&mut self` for
// methods exposed to Python.  For the chainable builder methods we take
// `&self` and return a new LazyPipeline (one clone, but only of the Vec
// descriptor — not the underlying data).
// ---------------------------------------------------------------------------
#[pyclass(module = "rmath")]
#[derive(Clone, Debug)]
pub struct LazyPipeline {
    source: Source,
    steps: Vec<PipelineStep>,
}

// ---------------------------------------------------------------------------
// Internal (Rust-only) implementation
// ---------------------------------------------------------------------------
impl LazyPipeline {
    // --- Source helpers ---

    fn source_len(&self) -> usize {
        match &self.source {
            Source::Range { start, stop, step } => {
                let n = ((stop - start) / step).ceil();
                if n <= 0.0 { 0 } else { n as usize }
            }
            Source::Buffer(v) => v.len(),
        }
    }

    #[inline(always)]
    fn source_val(&self, idx: usize) -> f64 {
        match &self.source {
            Source::Range { start, step, .. } => start + idx as f64 * step,
            Source::Buffer(v) => v[idx],
        }
    }

    // --- Single-element step application ---
    //
    // Returns None if a filter rejected the value.
    #[inline(always)]
    fn apply_steps_scalar(&self, mut x: f64) -> Option<f64> {
        for step in &self.steps {
            match step {
                PipelineStep::Map(op) => {
                    x = apply_op(*op, x);
                }
                PipelineStep::FilterGreater(v) => {
                    if x <= *v { return None; }
                }
                PipelineStep::FilterLess(v) => {
                    if x >= *v { return None; }
                }
                PipelineStep::FilterFinite => {
                    if !x.is_finite() { return None; }
                }
            }
        }
        Some(x)
    }

    // --- Check whether the pipeline has any filters ---
    fn has_filters(&self) -> bool {
        self.steps.iter().any(|s| !matches!(s, PipelineStep::Map(_)))
    }

    // --- Collect all Map ops into a plain Vec for the SIMD chunk path ---
    fn map_ops(&self) -> Vec<Op> {
        self.steps.iter().filter_map(|s| {
            if let PipelineStep::Map(op) = s { Some(*op) } else { None }
        }).collect()
    }

    // -----------------------------------------------------------------------
    // collect_serial: materialise [start_idx, end_idx) into a Vec<f64>
    // -----------------------------------------------------------------------
    fn collect_serial(&self, start_idx: usize, end_idx: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(end_idx - start_idx);
        for i in start_idx..end_idx {
            if let Some(v) = self.apply_steps_scalar(self.source_val(i)) {
                out.push(v);
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // collect_parallel: split across Rayon threads then concatenate
    // -----------------------------------------------------------------------
    fn collect_parallel(&self, total_n: usize) -> Vec<f64> {
        let n_threads = rayon::current_num_threads();
        let chunk = (total_n + n_threads - 1) / n_threads;
        (0..n_threads)
            .into_par_iter()
            .flat_map(|t| {
                let start = t * chunk;
                let end = (start + chunk).min(total_n);
                if start < end { self.collect_serial(start, end) } else { vec![] }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // sum_serial: sum over [start_idx, end_idx).
    //
    // Fast path: if there are NO filters, we fill a CHUNK_SIZE buffer from
    // the source and apply ops in-place, then accumulate.  This allows the
    // compiler to auto-vectorise the inner loop.
    //
    // Slow path (tail or filtered): scalar application one element at a time.
    //
    // BUG FIX (was in original): the original code ran the chunk fast-path
    // (incrementing i to cover the chunked range) but then reset to scalar
    // processing starting at i=0, re-processing every element including those
    // already summed.  Fixed here: the tail loop continues from where i left
    // off after the chunk loop.
    // -----------------------------------------------------------------------
    fn sum_serial(&self, start_idx: usize, end_idx: usize) -> f64 {
        let n = end_idx - start_idx;
        let mut total = 0.0_f64;
        let mut i = 0_usize;

        if !self.has_filters() {
            let ops = self.map_ops();
            // Process aligned chunks
            while i + CHUNK_SIZE <= n {
                let mut buf = [0.0_f64; CHUNK_SIZE];
                let base = start_idx + i;
                // Fill buffer from source
                match &self.source {
                    Source::Range { start, step, .. } => {
                        for j in 0..CHUNK_SIZE {
                            buf[j] = start + (base + j) as f64 * step;
                        }
                    }
                    Source::Buffer(v) => {
                        buf.copy_from_slice(&v[base..base + CHUNK_SIZE]);
                    }
                }
                // Apply all map ops to the chunk
                for &op in &ops {
                    for x in buf.iter_mut() {
                        *x = apply_op(op, *x);
                    }
                }
                // Accumulate (let the compiler vectorise this)
                for x in buf {
                    total += x;
                }
                i += CHUNK_SIZE;
            }
        }

        // Tail (or full scalar path if filters present)
        while i < n {
            if let Some(v) = self.apply_steps_scalar(self.source_val(start_idx + i)) {
                total += v;
            }
            i += 1;
        }
        total
    }

    fn max_serial(&self, start_idx: usize, end_idx: usize) -> f64 {
        let mut acc = f64::NEG_INFINITY;
        for i in start_idx..end_idx {
            if let Some(v) = self.apply_steps_scalar(self.source_val(i)) {
                if v > acc { acc = v; }
            }
        }
        acc
    }

    fn min_serial(&self, start_idx: usize, end_idx: usize) -> f64 {
        let mut acc = f64::INFINITY;
        for i in start_idx..end_idx {
            if let Some(v) = self.apply_steps_scalar(self.source_val(i)) {
                if v < acc { acc = v; }
            }
        }
        acc
    }

    // -----------------------------------------------------------------------
    // Welford online mean — single pass, O(1) memory, numerically stable.
    // Used for mean/std/var when filters are present (count unknown up front).
    // -----------------------------------------------------------------------
    fn welford_serial(&self, start_idx: usize, end_idx: usize) -> (f64, f64, usize) {
        // Returns (mean, M2, count) where variance = M2 / count.
        let mut mean = 0.0_f64;
        let mut m2 = 0.0_f64;
        let mut count = 0_usize;
        for i in start_idx..end_idx {
            if let Some(x) = self.apply_steps_scalar(self.source_val(i)) {
                count += 1;
                let delta = x - mean;
                mean += delta / count as f64;
                let delta2 = x - mean;
                m2 += delta * delta2;
            }
        }
        (mean, m2, count)
    }

    // Merge two Welford accumulators (for parallel reduction)
    fn welford_merge(
        (ma, m2a, na): (f64, f64, usize),
        (mb, m2b, nb): (f64, f64, usize),
    ) -> (f64, f64, usize) {
        if na == 0 { return (mb, m2b, nb); }
        if nb == 0 { return (ma, m2a, na); }
        let n = na + nb;
        let delta = mb - ma;
        let mean = ma + delta * (nb as f64 / n as f64);
        let m2 = m2a + m2b + delta * delta * (na as f64 * nb as f64 / n as f64);
        (mean, m2, n)
    }
}

// ---------------------------------------------------------------------------
// Free function: apply a single Op to a single f64.
// Kept outside impl so the SIMD chunk loop can call it without self.
// ---------------------------------------------------------------------------
#[inline(always)]
fn apply_op(op: Op, x: f64) -> f64 {
    match op {
        Op::Sin    => x.sin(),
        Op::Cos    => x.cos(),
        Op::Sqrt   => x.sqrt(),
        Op::Abs    => x.abs(),
        Op::Exp    => x.exp(),
        Op::Add(v) => x + v,
        Op::Mul(v) => x * v,
        Op::Sub(v) => x - v,
        Op::Div(v) => x / v,
    }
}

// ---------------------------------------------------------------------------
// Python-exposed methods
// ---------------------------------------------------------------------------
#[pymethods]
impl LazyPipeline {

    // -----------------------------------------------------------------------
    // Terminal operations — these trigger actual computation
    // -----------------------------------------------------------------------

    /// Materialise the pipeline into a `Vector`.
    ///
    /// Triggers parallel execution via Rayon if the source is large.
    ///
    /// Examples:
    ///     >>> from rmath import loop_range
    ///     >>> v = loop_range(10).sin().to_vector()
    ///     >>> type(v)
    ///     <class 'rmath.Vector'>
    pub fn to_vector(&self) -> Vector {
        let n = self.source_len();
        let data = if n < PARALLEL_THRESHOLD {
            self.collect_serial(0, n)
        } else {
            self.collect_parallel(n)
        };
        Vector::new(data)
    }

    /// Materialise the pipeline into a Python tuple of raw floats.
    ///
    /// This is the preferred way to iterate over pipeline results in a
    /// Python `for` loop.  Unlike wrapping each element in a `Scalar`,
    /// the tuple contains plain Python `float` objects — dramatically
    /// cheaper to construct (CPython float cache) and iterate over.
    ///
    /// ```python
    /// for val in sc.loop_range(10).sin().to_tuple():
    ///     print(val)   # val is a plain Python float
    /// ```
    pub fn to_tuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let n = self.source_len();
        let data = if n < PARALLEL_THRESHOLD {
            self.collect_serial(0, n)
        } else {
            self.collect_parallel(n)
        };
        pyo3::types::PyTuple::new(py, &data)
    }

    /// Sum all elements in the pipeline.
    ///
    /// Triggers immediate evaluation. Returns `0.0` for an empty pipeline.
    ///
    /// Examples:
    ///     >>> from rmath import loop_range
    ///     >>> loop_range(1, 4).sum()
    ///     6.0
    pub fn sum(&self) -> f64 {
        let n = self.source_len();
        if n == 0 { return 0.0; }
        if n < PARALLEL_THRESHOLD {
            self.sum_serial(0, n)
        } else {
            let n_threads = rayon::current_num_threads();
            let chunk = (n + n_threads - 1) / n_threads;
            (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let s = t * chunk;
                    let e = (s + chunk).min(n);
                    if s < e { self.sum_serial(s, e) } else { 0.0 }
                })
                .sum()
        }
    }

    /// Arithmetic mean.  Returns `NaN` for an empty pipeline.
    ///
    /// Uses Welford's online algorithm for numerical stability — single pass,
    /// no intermediate allocation regardless of filter presence.
    pub fn mean(&self) -> f64 {
        let n = self.source_len();
        if n == 0 { return f64::NAN; }

        if n < PARALLEL_THRESHOLD {
            let (m, _, count) = self.welford_serial(0, n);
            if count == 0 { f64::NAN } else { m }
        } else {
            let n_threads = rayon::current_num_threads();
            let chunk = (n + n_threads - 1) / n_threads;
            let acc = (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let s = t * chunk;
                    let e = (s + chunk).min(n);
                    if s < e { self.welford_serial(s, e) } else { (0.0, 0.0, 0) }
                })
                .reduce(|| (0.0, 0.0, 0), Self::welford_merge);
            if acc.2 == 0 { f64::NAN } else { acc.0 }
        }
    }

    /// Population variance (divided by N).
    ///
    /// Triggers immediate evaluation. Returns `NaN` for an empty pipeline.
    ///
    /// Examples:
    ///     >>> from rmath import loop_range
    ///     >>> loop_range(5).var()
    ///     2.0
    pub fn var(&self) -> f64 {
        let n = self.source_len();
        if n == 0 { return f64::NAN; }

        if n < PARALLEL_THRESHOLD {
            let (_, m2, count) = self.welford_serial(0, n);
            if count == 0 { f64::NAN } else { m2 / count as f64 }
        } else {
            let n_threads = rayon::current_num_threads();
            let chunk = (n + n_threads - 1) / n_threads;
            let acc = (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let s = t * chunk;
                    let e = (s + chunk).min(n);
                    if s < e { self.welford_serial(s, e) } else { (0.0, 0.0, 0) }
                })
                .reduce(|| (0.0, 0.0, 0), Self::welford_merge);
            if acc.2 == 0 { f64::NAN } else { acc.1 / acc.2 as f64 }
        }
    }

    /// Population standard deviation.
    ///
    /// Examples:
    ///     >>> from rmath import loop_range
    ///     >>> loop_range(5).std()
    ///     1.4142135623730951
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Maximum value in the pipeline.
    ///
    /// Returns `NaN` for an empty pipeline.
    ///
    /// Examples:
    ///     >>> from rmath import loop_range
    ///     >>> loop_range(10).max()
    ///     9.0
    pub fn max(&self) -> f64 {
        let n = self.source_len();
        if n == 0 { return f64::NAN; }
        if n < PARALLEL_THRESHOLD {
            self.max_serial(0, n)
        } else {
            let n_threads = rayon::current_num_threads();
            let chunk = (n + n_threads - 1) / n_threads;
            (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let s = t * chunk;
                    let e = (s + chunk).min(n);
                    if s < e { self.max_serial(s, e) } else { f64::NEG_INFINITY }
                })
                .reduce(|| f64::NEG_INFINITY, |a, b| a.max(b))
        }
    }

    /// Minimum value.  Returns `NaN` for an empty pipeline.
    pub fn min(&self) -> f64 {
        let n = self.source_len();
        if n == 0 { return f64::NAN; }
        if n < PARALLEL_THRESHOLD {
            self.min_serial(0, n)
        } else {
            let n_threads = rayon::current_num_threads();
            let chunk = (n + n_threads - 1) / n_threads;
            (0..n_threads)
                .into_par_iter()
                .map(|t| {
                    let s = t * chunk;
                    let e = (s + chunk).min(n);
                    if s < e { self.min_serial(s, e) } else { f64::INFINITY }
                })
                .reduce(|| f64::INFINITY, |a, b| a.min(b))
        }
    }

    // -----------------------------------------------------------------------
    // Builder / chaining methods
    //
    // Each method takes &self, clones the descriptor (cheap — only the Vec of
    // PipelineStep enums, not the source data), appends a step, and returns
    // the new pipeline.  The source Arc<Vec<f64>> is reference-counted, so
    // cloning a Buffer pipeline is O(1).
    // -----------------------------------------------------------------------

    /// Apply sine element-wise.
    pub fn sin(&self) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Sin));
        p
    }

    /// Apply cosine element-wise.
    pub fn cos(&self) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Cos));
        p
    }

    /// Apply square root element-wise.
    pub fn sqrt(&self) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Sqrt));
        p
    }

    /// Apply absolute value element-wise.
    pub fn abs(&self) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Abs));
        p
    }

    /// Apply exponential element-wise.
    pub fn exp(&self) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Exp));
        p
    }

    /// Add a scalar value to every element.
    pub fn add(&self, val: f64) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Add(val)));
        p
    }

    /// Multiply every element by a scalar value.
    pub fn mul(&self, val: f64) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Mul(val)));
        p
    }

    /// Subtract a scalar value from every element.
    pub fn sub(&self, val: f64) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Sub(val)));
        p
    }

    /// Divide every element by a scalar value.
    ///
    /// Raises `ZeroDivisionError` if `val == 0`.
    pub fn div(&self, val: f64) -> PyResult<LazyPipeline> {
        if val == 0.0 {
            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                "LazyPipeline.div: divisor cannot be zero",
            ));
        }
        let mut p = self.clone();
        p.steps.push(PipelineStep::Map(Op::Div(val)));
        Ok(p)
    }

    /// Keep only elements greater than `val`.
    pub fn filter_gt(&self, val: f64) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::FilterGreater(val));
        p
    }

    /// Keep only elements less than `val`.
    pub fn filter_lt(&self, val: f64) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::FilterLess(val));
        p
    }

    /// Remove NaN and infinite values.
    pub fn filter_finite(&self) -> LazyPipeline {
        let mut p = self.clone();
        p.steps.push(PipelineStep::FilterFinite);
        p
    }

    /// Type cast — currently all internal values are f64, so this is a
    /// documented no-op at the value level.  Retained for API compatibility
    /// and future tagged-union extension.
    pub fn as_(&self, _target: &str) -> LazyPipeline {
        self.clone()
    }

    // -----------------------------------------------------------------------
    // Python sequence protocol
    // -----------------------------------------------------------------------

    /// `len(pipeline)` — returns the *source* length before filters.
    /// After filters the count is unknown without evaluation.
    pub fn __len__(&self) -> usize {
        self.source_len()
    }

    pub fn __repr__(&self) -> String {
        let src = match &self.source {
            Source::Range { start, stop, step } => {
                format!("Range({start}, {stop}, step={step})")
            }
            Source::Buffer(v) => format!("Buffer(len={})", v.len()),
        };
        format!("LazyPipeline(source={src}, steps={})", self.steps.len())
    }
}

// ---------------------------------------------------------------------------
// Public Rust API (not #[pyfunction] — used by Vector and other Rust modules)
// ---------------------------------------------------------------------------

/// Create a pipeline from a shared immutable buffer.
/// Zero-copy: the Arc clone increments the reference count only.
pub fn from_shared_buffer(buffer: Arc<Vec<f64>>) -> LazyPipeline {
    LazyPipeline {
        source: Source::Buffer(buffer),
        steps: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Python-exposed free functions
// ---------------------------------------------------------------------------

/// Create a lazy range pipeline.
///
/// Returns values from `start` up to (but not including) `stop`.
/// Default `start` is 0.0, default `step` is 1.0.
///
/// Examples:
///     >>> from rmath import loop_range
///     >>> list(loop_range(5).to_tuple())
///     [0.0, 1.0, 2.0, 3.0, 4.0]
///     >>> list(loop_range(2, 5).to_tuple())
///     [2.0, 3.0, 4.0]
#[pyfunction]
#[pyo3(signature = (start, stop = None, step = 1.0))]
pub fn loop_range(start: f64, stop: Option<f64>, step: f64) -> PyResult<LazyPipeline> {
    if step == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "loop_range: step cannot be zero",
        ));
    }
    let (actual_start, actual_stop) = match stop {
        Some(s) => (start, s),
        None    => (0.0, start),
    };
    let n = ((actual_stop - actual_start) / step).ceil();
    if n < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "loop_range: step has the wrong sign for the given start/stop",
        ));
    }
    Ok(LazyPipeline {
        source: Source::Range {
            start: actual_start,
            stop:  actual_stop,
            step,
        },
        steps: Vec::new(),
    })
}

/// Create a pipeline from a Python list of floats.
///
/// Examples:
///     >>> from rmath import from_list
///     >>> from_list([1.0, 2.0, 3.0]).sum()
///     6.0
#[pyfunction]
pub fn from_list(data: Vec<f64>) -> LazyPipeline {
    LazyPipeline {
        source: Source::Buffer(Arc::new(data)),
        steps: Vec::new(),
    }
}

/// Create a pipeline of `n` zeros.
#[pyfunction]
pub fn zeros(n: usize) -> LazyPipeline {
    LazyPipeline {
        source: Source::Buffer(Arc::new(vec![0.0_f64; n])),
        steps: Vec::new(),
    }
}

/// Evenly spaced values from `start` to `stop` (inclusive), with `n` points.
///
/// Examples:
///     >>> from rmath import linspace
///     >>> list(linspace(0, 1, 5).to_tuple())
///     [0.0, 0.25, 0.5, 0.75, 1.0]
#[pyfunction]
pub fn linspace(start: f64, stop: f64, n: usize) -> PyResult<LazyPipeline> {
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "linspace: n must be >= 1",
        ));
    }
    if n == 1 {
        return Ok(LazyPipeline {
            source: Source::Buffer(Arc::new(vec![start])),
            steps: Vec::new(),
        });
    }
    let step = (stop - start) / (n - 1) as f64;
    Ok(LazyPipeline {
        source: Source::Range { start, stop: stop + step * 0.5, step },
        steps: Vec::new(),
    })
}
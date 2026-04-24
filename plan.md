# Documentation Strategy Plan: Rmath v0.1.5+

This document outlines the systematic approach to hardening the documentation across the Rmath ecosystem, ensuring parity between the Rust implementation, Python type stubs, and the web-based documentation portal.

## 1. Objectives
- **Full Parity**: Every public method in Rust must be documented in Python's `help()` and reflected in `.pyi` stubs.
- **Deep Context**: Document the new "Projected Storage" and "Lazy Evaluation" paradigms for Vector operations.
- **Visual Evidence**: Update the Portal with the latest 2026 performance benchmarks (Vector vs NumPy, Tensor vs PyTorch).
- **IDE Intelligence**: Ensure VS Code/PyCharm provide 100% accurate autocompletion and type-checking.

## 2. Component Breakdown

### A. Internal Docstrings (Rust Source)
We will audit `src/vector/core.rs`, `src/vector/ops.rs`, `src/array/core.rs`, and others to ensure:
- Detailed descriptions of "Materialization Protection" (the 250M element limit).
- Notes on thread-safety and GIL-releasing behavior for each method.
- Usage examples within the docstrings themselves for REPL visibility.

### B. Python Type Stubs (`.pyi` Files) ✅ COMPLETED
A methodical sweep of all `rmath/**/*.pyi` files to:
- ~~Add missing methods (e.g., `MmapArray.load_rows`, `Vector.mean`, `Vector.standardize`).~~
- ~~Use precise type hints (`npt.NDArray`, `typing.Iterable`, `PyResult` equivalents).~~
- ~~Document the `Projected` state of Vectors in comments within the stubs.~~

**Result:** Every public method across all 6 stub files (vector, array, scalar, calculus, linalg, stats) now has a docstring. Zero bare stubs remain.

### C. Documentation Portal (`docs/portal/`) ✅ DONE
Major updates to the static HTML portal:
- ~~**Foundations Page**: Explain the 3-tier memory model (Inline -> Heap -> Projected).~~
- ~~**Vector Page**: Highlight the "Impossible Vector" capability (50B+ elements).~~
- ~~**Tensor Page**: Update the PyTorch comparison benchmarks (reflecting the ~4.14x average speedup).~~
- ~~**Theory Page**: Detailed explanation of Kahan Summation and Welford's Algorithm used in our reductions.~~
- ~~**Benchmarks Page**: Add real-world pipeline comparison (rmath vs NumPy, 5M rows).~~

**Result:** All portal pages updated with semantic `<section>` layouts and deep architectural context.

## 3. Targeted Modules for Audit
1.  **Vector**: `core.rs`, `ops.rs`.
2.  **Array**: `core.rs`, `lazy.rs` (Focus on out-of-core Mmap and Lazy Fusion).
3.  **Tensor**: `autograd.rs` (Focus on Gradient tracking and Backprop).
4.  **Special**: `special.rs` (Gamma, Erf, Beta functions).

## 4. Implementation Workflow (Next Steps)
1.  **Phase 1**: Update all Rust docstrings to ensure `help()` is comprehensive.
2.  ~~**Phase 2**: Rebuild stubs using a combination of manual entry and inspection scripts to ensure 100% coverage.~~ ✅ DONE
3.  ~~**Phase 3**: Inject industrial-grade examples and benchmark data into the Portal HTML files.~~ ✅ DONE (benchmarks page)
4.  ~~**Phase 4**: Final audit against `benchmarks/*.py` to ensure all documented behavior matches reality.~~ ✅ DONE — pipeline scripts published to `benchmarks/pipeline/`

## 5. Additional Completed Work
- **README.md**: Added real-world pipeline benchmark table with time + memory columns.
- **benchmarks/pipeline/**: Created `rmath_pipeline.py` and `numpy_pipeline.py` for reproducible public benchmarks.

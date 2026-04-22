pub mod descriptive;
pub mod inferential;
pub mod regression;
pub mod distributions;

use pyo3::prelude::*;

pub fn register_stats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", "rmath.stats — high-performance parallelized statistical analysis.

This module provides a comprehensive suite of statistical tools, including
descriptive statistics, inferential tests, regression analysis, and 
probability distributions.

Key Features:
    1. Parallel Reductions: Statistical moments (mean, variance, skewness, 
       kurtosis) are calculated in a single parallel pass using Rayon.
    2. Numerical Stability: Uses Welford's online algorithm to prevent 
       precision loss during large-scale variance calculations.
    3. Distribution Kernels: High-performance probability density and 
       cumulative distribution functions backed by the `statrs` engine.

Examples:
    >>> import rmath.stats as rs
    >>> data = [1.2, 2.3, 1.8, 4.5, 3.1]
    >>> rs.describe(data)
    {'count': 5.0, 'mean': 2.58, ...}
")?;

    // Descriptive Statistics
    m.add_function(wrap_pyfunction!(descriptive::mean, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::variance, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::median, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::mode, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::describe, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::skewness, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::iqr, m)?)?;
    m.add_function(wrap_pyfunction!(descriptive::mad, m)?)?;
    
    // Inferential Statistics
    m.add_function(wrap_pyfunction!(inferential::correlation, m)?)?;
    m.add_function(wrap_pyfunction!(inferential::correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(inferential::covariance, m)?)?;
    m.add_function(wrap_pyfunction!(inferential::t_test_independent, m)?)?;
    // Alias for backward compatibility
    m.add("t_test", m.getattr("t_test_independent")?)?;
    m.add_function(wrap_pyfunction!(inferential::spearman_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(inferential::anova_oneway, m)?)?;
    
    // Regression
    m.add_function(wrap_pyfunction!(regression::linear_regression, m)?)?;

    // Distributions
    distributions::register_distributions(m)?;
    
    Ok(())
}

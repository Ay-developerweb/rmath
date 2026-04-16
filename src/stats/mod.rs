pub mod descriptive;
pub mod inferential;
pub mod regression;
pub mod distributions;

use pyo3::prelude::*;

pub fn register_stats(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

"""
rmath.stats — Industrial-strength Statistical Analysis.

Powered by a parallelized Rust engine for high-throughput computation.
Features include:
    - Descriptive stats: mean, variance, std_dev, skewness, kurtosis.
    - Inferential engines: t-test, f-test, and p-value approximations.
    - Vectorized operations for large-scale data processing.
"""
from .._rmath import stats as _stats

for name in dir(_stats):
    if not name.startswith('_'):
        globals()[name] = getattr(_stats, name)

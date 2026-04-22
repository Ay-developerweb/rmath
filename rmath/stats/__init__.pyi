"""
rmath.stats — Industrial-strength statistical analysis for Python.

This module provides high-performance, parallelized statistical tools for
descriptive analysis, inferential testing, regression, and probability 
distributions. All heavy computations are offloaded to Rust and 
automatically use the Rayon thread pool for data larger than 8,192 elements.

Numerical stability is maintained via Welford's online algorithm for 
moments and Kahan summation for accumulators.
"""

from typing import List, Sequence, Union, Tuple, Dict, Any, Optional
from rmath.vector import Vector
from rmath.array import Array

# --- Descriptive Statistics ---

def mean(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the arithmetic mean. Returns `NaN` for empty data.
    
    Examples:
        >>> from rmath.stats import mean
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    ...

def median(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the median value.
    
    Implementation: Uses an unstable O(N) selection algorithm (introselect)
    which is significantly faster than sorting the entire array.
    """
    ...

def mode(data: Union[Sequence[float], Vector]) -> float:
    """Find the most common value. 
    
    In case of ties (multiple modes), returns the one that appeared first 
    in the input sequence.
    """
    ...

def describe(data: Union[Sequence[float], Vector]) -> Dict[str, float]:
    """Generate a comprehensive summary of descriptive statistics.
    
    Returns:
        A dictionary containing:
        - 'count': Number of observations
        - 'mean': Arithmetic average
        - 'variance': Sample variance (n-1)
        - 'std': Sample standard deviation
        - 'skewness': Fisher-Pearson skewness
        - 'kurtosis': Excess kurtosis (Fisher definition)
    
    Performance:
        Calculates all moments in a single parallel pass using Welford's 
        algorithm, ensuring O(N) time and O(1) extra space.
    """
    ...

def variance(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the sample variance (degree of freedom = 1)."""
    ...

def std_dev(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the sample standard deviation."""
    ...

def skewness(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the Fisher-Pearson coefficient of skewness."""
    ...

def kurtosis(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the excess kurtosis (Fisher definition)."""
    ...

def iqr(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the Interquartile Range (75th - 25th percentile)."""
    ...

def mad(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the Median Absolute Deviation (MAD)."""
    ...

# --- Inferential Statistics ---

def correlation(x: Union[Sequence[float], Vector], y: Union[Sequence[float], Vector]) -> float:
    """Calculate the Pearson correlation coefficient (r) between x and y."""
    ...

def correlation_matrix(arr: Array) -> Array:
    """Calculate the Pearson correlation matrix for an Array.
    
    Expects rows as variables and columns as observations.
    """
    ...

def covariance(x: Union[Sequence[float], Vector], y: Union[Sequence[float], Vector]) -> float:
    """Calculate the sample covariance between x and y."""
    ...

def t_test_independent(a: Union[Sequence[float], Vector], b: Union[Sequence[float], Vector]) -> Tuple[float, float]:
    """Perform Welch's T-test for two independent samples.
    
    Returns:
        (t_statistic, p_value)
    """
    ...

def t_test(a: Union[Sequence[float], Vector], b: Union[Sequence[float], Vector]) -> Tuple[float, float]:
    """Alias for `t_test_independent`."""
    ...

def spearman_correlation(x: Union[Sequence[float], Vector], y: Union[Sequence[float], Vector]) -> float:
    """Calculate the Spearman rank correlation coefficient."""
    ...

def anova_oneway(groups: Sequence[Union[Sequence[float], Vector]]) -> Tuple[float, float, float]:
    """Perform a One-Way ANOVA across multiple groups.
    
    Returns:
        (f_statistic, df_between, df_within)
    """
    ...

# --- Regression ---

def linear_regression(x: Union[Sequence[float], Vector], y: Union[Sequence[float], Vector]) -> Dict[str, float]:
    """Perform simple linear regression using Ordinary Least Squares (OLS).
    
    Returns:
        Dict with 'slope', 'intercept', and 'r_squared'.
    """
    ...

# --- Distributions ---

class Normal:
    """Normal (Gaussian) distribution: N(μ, σ)."""
    def __init__(self, mu: float, sigma: float) -> None: ...
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, x: float) -> float: ...
    def mean(self) -> float: ...
    def sample(self, n: int) -> Vector: ...

class StudentT:
    """Student's T-distribution: t(location, scale, freedom)."""
    def __init__(self, location: float, scale: float, freedom: float) -> None: ...
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, x: float) -> float: ...
    def sample(self, n: int) -> Vector: ...

class Poisson:
    """Poisson distribution for discrete events: Pois(lambda)."""
    def __init__(self, lambda_: float) -> None: ...
    def pmf(self, k: int) -> float: ...
    def cdf(self, k: float) -> float: ...
    def sample(self, n: int) -> Vector: ...

class Exponential:
    """Exponential distribution (waiting times): Exp(rate)."""
    def __init__(self, rate: float) -> None: ...
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, x: float) -> float: ...
    def sample(self, n: int) -> Vector: ...

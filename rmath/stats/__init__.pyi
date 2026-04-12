"""
rmath.stats — Industrial-strength statistical analysis for Python.
All functions take list[float] or rmath.Vector and provide high-performance
parallel processing via Rayon.
"""

from typing import List, Sequence, Union, Tuple, Dict, Any
from rmath import Vector

# --- Descriptive Statistics ---

def sum(data: Union[Sequence[float], Vector]) -> float:
    """Return the arithmetic sum of the dataset."""
    ...

def mean(data: Union[Sequence[float], Vector]) -> float:
    """Calculate the arithmetic mean. Returns NaN for empty data."""
    ...

def median(data: Union[Sequence[float], Vector]) -> float:
    """
    Calculate the median.
    Uses an unstable O(N) selection algorithm for high performance.
    """
    ...

def mode(data: Union[Sequence[float], Vector]) -> float:
    """Find the most common value. In case of ties, returns the first encountered."""
    ...

def describe(data: Union[Sequence[float], Vector]) -> Dict[str, float]:
    """
    Generate a summary report of descriptive statistics.
    Returns:
        Dict containing 'count', 'mean', 'variance', 'std', 'skewness', 'kurtosis'.
    Uses parallelized Welford's algorithm for numerical stability.
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
    """Calculate the Pearson correlation coefficient between two variables."""
    ...

def covariance(x: Union[Sequence[float], Vector], y: Union[Sequence[float], Vector]) -> float:
    """Calculate the sample covariance between x and y."""
    ...

def t_test_independent(a: Union[Sequence[float], Vector], b: Union[Sequence[float], Vector]) -> Tuple[float, float]:
    """
    Perform Welch's T-test for two independent samples.
    Returns:
        (t_statistic, p_value)
    """
    ...

def spearman_correlation(x: Union[Sequence[float], Vector], y: Union[Sequence[float], Vector]) -> float:
    """Calculate the Spearman rank correlation coefficient."""
    ...

def anova_oneway(groups: Sequence[Union[Sequence[float], Vector]]) -> Tuple[float, float, float]:
    """
    Perform a One-Way ANOVA across multiple groups.
    Returns:
        (f_statistic, df_between, df_within)
    """
    ...

# --- Regression ---

def linear_regression(x: Union[Sequence[float], Vector], y: Union[Sequence[float], Vector]) -> Dict[str, float]:
    """
    Perform simple linear regression.
    Returns:
        Dict with 'slope', 'intercept', 'r_squared'.
    """
    ...

# --- Distributions ---

class Normal:
    """Normal (Gaussian) distribution."""
    def __init__(self, mu: float, sigma: float) -> None: ...
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, x: float) -> float: ...
    def mean(self) -> float: ...

class StudentT:
    """Student's T-distribution."""
    def __init__(self, location: float, scale: float, freedom: float) -> None: ...
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, x: float) -> float: ...

class Poisson:
    """Poisson distribution for discrete events."""
    def __init__(self, lambda_: float) -> None: ...
    def pmf(self, k: int) -> float: ...
    def cdf(self, k: float) -> float: ...

class Exponential:
    """Exponential distribution (waiting times)."""
    def __init__(self, rate: float) -> None: ...
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, x: float) -> float: ...

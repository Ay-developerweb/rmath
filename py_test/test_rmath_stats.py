# py_test/test_rmath_stats.py

import math
import statistics
import time

import numpy as np
import pytest

import rmath.stats as rs
from rmath import Vector


# ── helpers ──────────────────────────────────────────────────────────────────

def bench(label: str, fn, *args, iterations: int = 200_000):
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    elapsed = time.perf_counter() - t0
    print(f"  {label:45s} {elapsed:.3f}s ({iterations:,} iters)")


# ═══════════════════════════════════════════════════════════════════════════
# 1. sum
# ═══════════════════════════════════════════════════════════════════════════

class TestSum:
    def test_basic(self):
        assert rs.sum([1.0, 2.0, 3.0]) == 6.0

    def test_single(self):
        assert rs.sum([42.0]) == 42.0

    def test_empty(self):
        assert rs.sum([]) == 0.0

    def test_negative(self):
        assert rs.sum([-1.0, -2.0, -3.0]) == pytest.approx(-6.0)

    def test_vector(self):
        assert rs.sum(Vector([1.0, 2.0, 3.0])) == 6.0

    def test_matches_builtin(self):
        data = [float(i) for i in range(1, 101)]
        assert rs.sum(data) == pytest.approx(sum(data))


# ═══════════════════════════════════════════════════════════════════════════
# 2. mean
# ═══════════════════════════════════════════════════════════════════════════

class TestMean:
    def test_basic(self):
        assert rs.mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single(self):
        assert rs.mean([7.0]) == pytest.approx(7.0)

    def test_negative(self):
        assert rs.mean([-3.0, -1.0, -2.0]) == pytest.approx(-2.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="mean requires at least one element"):
            rs.mean([])

    def test_vector(self):
        assert rs.mean(Vector([1.0, 2.0, 3.0])) == pytest.approx(2.0)

    def test_matches_statistics(self):
        data = [1.5, 2.5, 3.5, 4.5]
        assert rs.mean(data) == pytest.approx(statistics.mean(data))


# ═══════════════════════════════════════════════════════════════════════════
# 3. variance
# ═══════════════════════════════════════════════════════════════════════════

class TestVariance:
    def test_basic(self):
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert rs.variance(data) == pytest.approx(statistics.variance(data))

    def test_two_elements(self):
        assert rs.variance([0.0, 2.0]) == pytest.approx(2.0)

    def test_single_raises(self):
        with pytest.raises(ValueError, match="variance requires at least two elements"):
            rs.variance([1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="variance requires at least two elements"):
            rs.variance([])

    def test_vector(self):
        data = [1.0, 2.0, 3.0, 4.0]
        assert rs.variance(Vector(data)) == pytest.approx(statistics.variance(data))

    def test_matches_statistics(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert rs.variance(data) == pytest.approx(statistics.variance(data))


# ═══════════════════════════════════════════════════════════════════════════
# 4. std_dev
# ═══════════════════════════════════════════════════════════════════════════

class TestStdDev:
    def test_basic(self):
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        assert rs.std_dev(data) == pytest.approx(statistics.stdev(data))

    def test_roundtrip_variance(self):
        data = [1.0, 3.0, 5.0, 7.0, 9.0]
        assert rs.std_dev(data) ** 2 == pytest.approx(rs.variance(data))

    def test_single_raises(self):
        with pytest.raises(ValueError, match="variance requires at least two elements"):
            rs.std_dev([1.0])

    def test_vector(self):
        data = [1.0, 2.0, 3.0]
        assert rs.std_dev(Vector(data)) == pytest.approx(statistics.stdev(data))

    def test_matches_statistics(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert rs.std_dev(data) == pytest.approx(statistics.stdev(data))


# ═══════════════════════════════════════════════════════════════════════════
# 5. geometric_mean
# ═══════════════════════════════════════════════════════════════════════════

class TestGeometricMean:
    def test_basic(self):
        # geometric_mean(1, 4, 16) = (1*4*16)^(1/3) = 64^(1/3) = 4
        assert rs.geometric_mean([1.0, 4.0, 16.0]) == pytest.approx(4.0)

    def test_single(self):
        assert rs.geometric_mean([9.0]) == pytest.approx(9.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="geometric_mean requires at least one element"):
            rs.geometric_mean([])

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="geometric_mean requires all values to be positive"):
            rs.geometric_mean([1.0, 0.0, 2.0])

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="geometric_mean requires all values to be positive"):
            rs.geometric_mean([1.0, -1.0, 2.0])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="geometric_mean does not support NaN"):
            rs.geometric_mean([1.0, float("nan"), 2.0])

    def test_vector(self):
        assert rs.geometric_mean(Vector([1.0, 4.0, 16.0])) == pytest.approx(4.0)

    def test_matches_statistics(self):
        data = [1.0, 2.0, 4.0, 8.0]
        assert rs.geometric_mean(data) == pytest.approx(statistics.geometric_mean(data))


# ═══════════════════════════════════════════════════════════════════════════
# 6. harmonic_mean
# ═══════════════════════════════════════════════════════════════════════════

class TestHarmonicMean:
    def test_basic(self):
        # harmonic_mean(1, 2, 4) = 3 / (1 + 0.5 + 0.25) = 12/7
        assert rs.harmonic_mean([1.0, 2.0, 4.0]) == pytest.approx(12.0 / 7.0)

    def test_single(self):
        assert rs.harmonic_mean([5.0]) == pytest.approx(5.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="harmonic_mean requires at least one element"):
            rs.harmonic_mean([])

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="harmonic_mean requires all values to be non-zero"):
            rs.harmonic_mean([1.0, 0.0, 2.0])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="harmonic_mean does not support NaN"):
            rs.harmonic_mean([1.0, float("nan")])

    def test_vector(self):
        assert rs.harmonic_mean(Vector([1.0, 2.0, 4.0])) == pytest.approx(12.0 / 7.0)

    def test_matches_statistics(self):
        data = [2.5, 3.0, 10.0]
        assert rs.harmonic_mean(data) == pytest.approx(statistics.harmonic_mean(data))


# ═══════════════════════════════════════════════════════════════════════════
# 7. median
# ═══════════════════════════════════════════════════════════════════════════

class TestMedian:
    def test_odd(self):
        assert rs.median([3.0, 1.0, 2.0]) == pytest.approx(2.0)

    def test_even(self):
        assert rs.median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_single(self):
        assert rs.median([99.0]) == pytest.approx(99.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="median requires at least one element"):
            rs.median([])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="median does not support NaN"):
            rs.median([1.0, float("nan"), 3.0])

    def test_sorted_input(self):
        assert rs.median([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.0)

    def test_reverse_sorted(self):
        assert rs.median([5.0, 4.0, 3.0, 2.0, 1.0]) == pytest.approx(3.0)

    def test_vector(self):
        assert rs.median(Vector([1.0, 3.0, 2.0])) == pytest.approx(2.0)

    def test_matches_statistics(self):
        data = [5.0, 1.0, 4.0, 2.0, 3.0]
        assert rs.median(data) == pytest.approx(statistics.median(data))


# ═══════════════════════════════════════════════════════════════════════════
# 8. mode
# ═══════════════════════════════════════════════════════════════════════════

class TestMode:
    def test_basic(self):
        assert rs.mode([1.0, 2.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single(self):
        assert rs.mode([7.0]) == pytest.approx(7.0)

    def test_all_same(self):
        assert rs.mode([5.0, 5.0, 5.0]) == pytest.approx(5.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="mode requires at least one element"):
            rs.mode([])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="mode does not support NaN"):
            rs.mode([1.0, float("nan"), 1.0])

    def test_vector(self):
        assert rs.mode(Vector([1.0, 2.0, 2.0, 3.0])) == pytest.approx(2.0)

    def test_matches_statistics(self):
        data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        assert rs.mode(data) == pytest.approx(statistics.mode(data))


# ═══════════════════════════════════════════════════════════════════════════
# 9. correlation
# ═══════════════════════════════════════════════════════════════════════════

class TestCorrelation:
    def test_perfect_positive(self):
        assert rs.correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert rs.correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_zero_correlation(self):
        # constant y → zero denominator → 0.0 by convention
        assert rs.correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            rs.correlation([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least two elements"):
            rs.correlation([1.0], [1.0])

    def test_vector(self):
        r = rs.correlation(Vector([1.0, 2.0, 3.0]), Vector([2.0, 4.0, 6.0]))
        assert r == pytest.approx(1.0)

    def test_matches_numpy(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 5.0, 4.0, 5.0]
        assert rs.correlation(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])


# ═══════════════════════════════════════════════════════════════════════════
# 10. skewness
# ═══════════════════════════════════════════════════════════════════════════

class TestSkewness:
    def test_symmetric(self):
        assert rs.skewness([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.0, abs=1e-10)

    def test_right_skewed(self):
        assert rs.skewness([1.0, 1.0, 1.0, 1.0, 10.0]) > 0

    def test_left_skewed(self):
        assert rs.skewness([1.0, 10.0, 10.0, 10.0, 10.0]) < 0

    def test_constant_returns_zero(self):
        assert rs.skewness([3.0, 3.0, 3.0]) == pytest.approx(0.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="skewness requires at least three elements"):
            rs.skewness([1.0, 2.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="skewness requires at least three elements"):
            rs.skewness([])

    def test_vector(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert rs.skewness(Vector(data)) == pytest.approx(rs.skewness(data))

    def test_matches_statistics(self):
        # manually compute adjusted Fisher-Pearson using stdlib statistics
        data = [2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]
        n = len(data)
        m = statistics.mean(data)
        s = statistics.stdev(data)
        m3 = sum(((x - m) / s) ** 3 for x in data)
        expected = (n / ((n - 1) * (n - 2))) * m3
        assert rs.skewness(data) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# 11. z_scores
# ═══════════════════════════════════════════════════════════════════════════

class TestZScores:
    def test_basic(self):
        zs = rs.z_scores([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert pytest.approx(sum(zs), abs=1e-10) == 0.0

    def test_constant_returns_zeros(self):
        assert rs.z_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]

    def test_mean_zero(self):
        zs = rs.z_scores([1.0, 2.0, 3.0, 4.0, 5.0])
        assert pytest.approx(sum(zs), abs=1e-10) == 0.0

    def test_std_one(self):
        zs = rs.z_scores([1.0, 2.0, 3.0, 4.0, 5.0])
        sd = math.sqrt(sum(z ** 2 for z in zs) / (len(zs) - 1))
        assert sd == pytest.approx(1.0)

    def test_single_raises(self):
        with pytest.raises(ValueError, match="z_scores requires at least two elements"):
            rs.z_scores([1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="z_scores requires at least two elements"):
            rs.z_scores([])

    def test_vector(self):
        data = [1.0, 2.0, 3.0]
        assert rs.z_scores(Vector(data)) == pytest.approx(rs.z_scores(data))

    def test_matches_statistics(self):
        # ddof=1 z-scores computed using stdlib statistics
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        m = statistics.mean(data)
        s = statistics.stdev(data)  # ddof=1
        expected = [(x - m) / s for x in data]
        assert rs.z_scores(data) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmark:
    """Run with: pytest -s py_test/test_rmath_stats.py::TestBenchmark"""

    N = 1_000

    def setup_method(self):
        self.data = [float(i) for i in range(1, self.N + 1)]
        self.vec  = Vector(self.data)
        self.x    = self.data
        self.y    = list(reversed(self.data))

    def test_bench_sum(self):
        print(f"\nBenchmarks — N={self.N:,}")
        bench("rmath.stats.sum (list)",        rs.sum,          self.data)
        bench("rmath.stats.sum (Vector)",      rs.sum,          self.vec)
        bench("builtins.sum",                  sum,             self.data)

    def test_bench_mean(self):
        bench("rmath.stats.mean (list)",       rs.mean,         self.data)
        bench("rmath.stats.mean (Vector)",     rs.mean,         self.vec)
        bench("statistics.mean",               statistics.mean, self.data)

    def test_bench_variance(self):
        bench("rmath.stats.variance (list)",   rs.variance,          self.data)
        bench("rmath.stats.variance (Vector)", rs.variance,          self.vec)
        bench("statistics.variance",           statistics.variance,  self.data)

    def test_bench_std_dev(self):
        bench("rmath.stats.std_dev (list)",    rs.std_dev,       self.data)
        bench("rmath.stats.std_dev (Vector)",  rs.std_dev,       self.vec)
        bench("statistics.stdev",              statistics.stdev, self.data)

    def test_bench_geometric_mean(self):
        bench("rmath.stats.geometric_mean (list)",   rs.geometric_mean,         self.data)
        bench("rmath.stats.geometric_mean (Vector)", rs.geometric_mean,         self.vec)
        bench("statistics.geometric_mean",           statistics.geometric_mean, self.data)

    def test_bench_harmonic_mean(self):
        bench("rmath.stats.harmonic_mean (list)",   rs.harmonic_mean,         self.data)
        bench("rmath.stats.harmonic_mean (Vector)", rs.harmonic_mean,         self.vec)
        bench("statistics.harmonic_mean",           statistics.harmonic_mean, self.data)

    def test_bench_median(self):
        bench("rmath.stats.median (list)",     rs.median,         self.data)
        bench("rmath.stats.median (Vector)",   rs.median,         self.vec)
        bench("statistics.median",             statistics.median, self.data)

    def test_bench_mode(self):
        bench("rmath.stats.mode (list)",       rs.mode,         self.data)
        bench("rmath.stats.mode (Vector)",     rs.mode,         self.vec)
        bench("statistics.mode",               statistics.mode, self.data)

    def test_bench_correlation(self):
        bench("rmath.stats.correlation (list)",   rs.correlation, self.x,   self.y)
        bench("rmath.stats.correlation (Vector)", rs.correlation, self.vec, Vector(self.y))
        bench("numpy corrcoef (lambda overhead)", lambda: np.corrcoef(self.x, self.y)[0, 1])

    def test_bench_skewness(self):
        bench("rmath.stats.skewness (list)",   rs.skewness, self.data)
        bench("rmath.stats.skewness (Vector)", rs.skewness, self.vec)
        # no stdlib/numpy equivalent — omit reference bench

    def test_bench_z_scores(self):
        bench("rmath.stats.z_scores (list)",   rs.z_scores, self.data)
        bench("rmath.stats.z_scores (Vector)", rs.z_scores, self.vec)
        # no stdlib/numpy equivalent — omit reference bench
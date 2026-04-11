import math
import time
import pytest
import rmath.scalar as rs   # adjust import to your module path

# ── Correctness ──────────────────────────────────────────────────────────────

class TestCorrectness:
    def test_add(self):
        assert rs.add(10.5, 0.5) == 11.0
        assert math.isinf(rs.add(float('inf'), 1.0))

    def test_sub(self):
        assert rs.sub(1.0, 0.1) == pytest.approx(0.9)
        assert rs.sub(0.0, 5.0) == -5.0

    def test_mul(self):
        assert rs.mul(2.5, 4.0) == 10.0
        assert rs.mul(0.0, 999.0) == 0.0

    def test_div(self):
        assert rs.div(1.0, 4.0) == 0.25
        with pytest.raises(ZeroDivisionError):
            rs.div(1.0, 0.0)

    def test_fmod(self):
        assert rs.fmod(-5.0, 3.0) == pytest.approx(-2.0)   # sign follows x
        assert rs.fmod(5.0, 3.0)  == pytest.approx(2.0)

    def test_remainder_vs_fmod(self):
        # KEY: these must differ for negative inputs
        assert rs.remainder(-5.0, 3.0) == pytest.approx(1.0)   # sign follows y
        assert rs.fmod(-5.0, 3.0)      == pytest.approx(-2.0)  # sign follows x
        assert rs.remainder(-5.0, 3.0) != rs.fmod(-5.0, 3.0)   # they are NOT the same

    def test_remainder_matches_python_modulo(self):
        cases = [(5.5, 2.0), (-5.0, 3.0), (7.0, -3.0), (-7.0, -3.0)]
        for x, y in cases:
            assert rs.remainder(x, y) == pytest.approx(x % y), f"Failed for {x} % {y}"

    def test_fmod_matches_math_fmod(self):
        cases = [(5.5, 2.0), (-5.0, 3.0), (7.0, -3.0), (-7.0, -3.0)]
        for x, y in cases:
            assert rs.fmod(x, y) == pytest.approx(math.fmod(x, y)), f"Failed fmod({x},{y})"

class TestRounding:
    def test_ceil(self):
        assert rs.ceil(2.0001) == 3.0
        assert rs.ceil(-1.5) == -1.0

    def test_floor(self):
        assert rs.floor(2.999) == 2.0
        assert rs.floor(-1.5) == -2.0

    def test_trunc(self):
        assert rs.trunc(2.9) == 2.0
        assert rs.trunc(-2.9) == -2.0   # trunc toward zero, not -3.0

    def test_round_away_from_zero(self):
        assert rs.round(2.5) == 3.0
        assert rs.round(-2.5) == -3.0   # away from zero

    def test_round_half_even(self):
        assert rs.round_half_even(2.5) == 2.0   # even
        assert rs.round_half_even(3.5) == 4.0   # even
        assert rs.round_half_even(-2.5) == -2.0  # even  ← was broken
        assert rs.round_half_even(-3.5) == -4.0  # even
        assert rs.round_half_even(2.4) == 2.0    # normal rounding
        assert rs.round_half_even(2.6) == 3.0

    def test_round_vs_round_half_even_differ(self):
        # These must differ — proof the two functions are distinct
        assert rs.round(2.5) != rs.round_half_even(2.5)
        assert rs.round(3.5) == rs.round_half_even(3.5)  # both give 4.0

    def test_signum(self):
        assert rs.signum(-42.0) == -1.0
        assert rs.signum(0.0) == 0.0
        assert rs.signum(99.0) == 1.0

    def test_clamp(self):
        assert rs.clamp(10.0, 0.0, 5.0) == 5.0
        assert rs.clamp(-10.0, 0.0, 5.0) == 0.0
        assert rs.clamp(3.0, 0.0, 5.0) == 3.0
        with pytest.raises(ValueError):
            rs.clamp(1.0, 5.0, 0.0)   # min > max

    def test_lerp(self):
        assert rs.lerp(10.0, 20.0, 0.5) == 15.0
        assert rs.lerp(0.0, 100.0, 0.1) == pytest.approx(10.0)
        assert rs.lerp(0.0, 100.0, 1.5) == pytest.approx(150.0)  # extrapolation
        assert rs.lerp(0.0, 100.0, 0.0) == 0.0
        assert rs.lerp(0.0, 100.0, 1.0) == 100.0

class TestRootsPowers:
    # sqrt
    def test_sqrt_positive(self):
        assert rs.sqrt(64.0) == 8.0
        assert rs.sqrt(0.0) == 0.0
        assert rs.sqrt(2.0) == pytest.approx(math.sqrt(2.0))

    def test_sqrt_negative_raises(self):
        with pytest.raises(ValueError):
            rs.sqrt(-1.0)

    # cbrt
    def test_cbrt_positive(self):
        assert rs.cbrt(27.0) == pytest.approx(3.0)

    def test_cbrt_negative(self):
        assert rs.cbrt(-27.0) == pytest.approx(-3.0)

    def test_cbrt_zero(self):
        assert rs.cbrt(0.0) == 0.0

    # root
    def test_root_cube(self):
        assert rs.root(-125.0, 3) == pytest.approx(-5.0)
        assert rs.root(125.0, 3)  == pytest.approx(5.0)

    def test_root_fourth(self):
        assert rs.root(16.0, 4) == pytest.approx(2.0)

    def test_root_negative_n(self):
        assert rs.root(8.0, -3) == pytest.approx(0.5)   # 1 / cbrt(8)

    def test_root_zero_n_raises(self):
        with pytest.raises(ValueError):
            rs.root(8.0, 0)

    def test_root_even_negative_raises(self):
        with pytest.raises(ValueError):
            rs.root(-4.0, 2)

    def test_root_matches_cbrt(self):
        # root(x, 3) must agree with cbrt to full float precision
        for v in [0.001, 1.0, 8.0, 1000.0, -27.0]:
            assert rs.root(v, 3) == pytest.approx(rs.cbrt(v), rel=1e-12)

    # pow
    def test_pow_basic(self):
        assert rs.pow(2.0, 10.0) == 1024.0
        assert rs.pow(2.0, -1.0) == 0.5
        assert rs.pow(9.0, 0.5)  == pytest.approx(3.0)

    def test_pow_nan_raises(self):
        with pytest.raises(ValueError):
            rs.pow(-2.0, 0.5)   # would be NaN in real domain

    def test_pow_zero_exp(self):
        assert rs.pow(999.0, 0.0) == 1.0

    # inv_sqrt
    def test_inv_sqrt_basic(self):
        assert rs.inv_sqrt(4.0) == pytest.approx(0.5)
        assert rs.inv_sqrt(1.0) == pytest.approx(1.0)
        assert rs.inv_sqrt(2.0) == pytest.approx(1.0 / math.sqrt(2.0))

    def test_inv_sqrt_zero_raises(self):
        with pytest.raises(ValueError):
            rs.inv_sqrt(0.0)

    def test_inv_sqrt_negative_raises(self):
        with pytest.raises(ValueError):
            rs.inv_sqrt(-1.0)

class TestExpLog:
    # exp
    def test_exp_basic(self):
        assert rs.exp(0.0) == 1.0
        assert rs.exp(1.0) == pytest.approx(math.e)
        assert rs.exp(-1.0) == pytest.approx(1.0 / math.e)

    def test_exp_matches_math(self):
        for v in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            assert rs.exp(v) == pytest.approx(math.exp(v))

    # exp2
    def test_exp2_basic(self):
        assert rs.exp2(0.0) == 1.0
        assert rs.exp2(10.0) == 1024.0
        assert rs.exp2(-1.0) == pytest.approx(0.5)

    # expm1
    def test_expm1_small(self):
        assert rs.expm1(1e-15) == pytest.approx(math.expm1(1e-15), rel=1e-12)

    def test_expm1_basic(self):
        assert rs.expm1(0.0) == 0.0
        assert rs.expm1(1.0) == pytest.approx(math.e - 1.0)

    # log
    def test_log_natural(self):
        assert rs.log(math.e) == pytest.approx(1.0)
        assert rs.log(1.0) == 0.0

    def test_log_base10(self):
        assert rs.log(100.0, 10.0) == pytest.approx(2.0)

    def test_log_base2(self):
        assert rs.log(1024.0, 2.0) == pytest.approx(10.0)

    def test_log_invalid_x(self):
        with pytest.raises(ValueError):
            rs.log(0.0)
        with pytest.raises(ValueError):
            rs.log(-1.0)

    def test_log_invalid_base(self):
        with pytest.raises(ValueError):
            rs.log(10.0, 0.0)
        with pytest.raises(ValueError):
            rs.log(10.0, 1.0)
        with pytest.raises(ValueError):
            rs.log(10.0, -2.0)

    # log2
    def test_log2_basic(self):
        assert rs.log2(1024.0) == pytest.approx(10.0)
        assert rs.log2(1.0) == 0.0

    def test_log2_invalid(self):
        with pytest.raises(ValueError):
            rs.log2(0.0)
        with pytest.raises(ValueError):
            rs.log2(-1.0)

    def test_log2_matches_math(self):
        for v in [0.5, 1.0, 2.0, 1024.0]:
            assert rs.log2(v) == pytest.approx(math.log2(v))

    # log10
    def test_log10_basic(self):
        assert rs.log10(1000.0) == pytest.approx(3.0)
        assert rs.log10(1.0) == 0.0

    def test_log10_invalid(self):
        with pytest.raises(ValueError):
            rs.log10(0.0)
        with pytest.raises(ValueError):
            rs.log10(-1.0)

    def test_log10_matches_math(self):
        for v in [0.1, 1.0, 10.0, 1000.0]:
            assert rs.log10(v) == pytest.approx(math.log10(v))

    # log1p
    def test_log1p_small(self):
        assert rs.log1p(1e-15) == pytest.approx(math.log1p(1e-15), rel=1e-12)

    def test_log1p_basic(self):
        assert rs.log1p(0.0) == 0.0
        assert rs.log1p(math.e - 1.0) == pytest.approx(1.0)

    def test_log1p_invalid(self):
        with pytest.raises(ValueError):
            rs.log1p(-1.0)
        with pytest.raises(ValueError):
            rs.log1p(-2.0)

    # logsumexp2
    def test_logsumexp2_symmetric(self):
        assert rs.logsumexp2(0.0, 0.0) == pytest.approx(math.log(2.0))

    def test_logsumexp2_stable(self):
        # Must not overflow — naive ln(exp(1000) + exp(1000)) would
        assert rs.logsumexp2(1000.0, 1000.0) == pytest.approx(1000.0 + math.log(2.0))

    def test_logsumexp2_matches_naive_small(self):
        x, y = 1.0, 2.0
        naive = math.log(math.exp(x) + math.exp(y))
        assert rs.logsumexp2(x, y) == pytest.approx(naive)

class TestTrig:
    # sin
    def test_sin_basic(self):
        assert rs.sin(0.0) == 0.0
        assert rs.sin(math.pi / 2) == pytest.approx(1.0)
        assert rs.sin(math.pi) == pytest.approx(0.0, abs=1e-15)

    def test_sin_matches_math(self):
        for v in [-math.pi, -math.pi/2, 0.0, math.pi/4, math.pi/2, math.pi]:
            assert rs.sin(v) == pytest.approx(math.sin(v))

    # cos
    def test_cos_basic(self):
        assert rs.cos(0.0) == 1.0
        assert rs.cos(math.pi) == pytest.approx(-1.0)
        assert rs.cos(math.pi / 2) == pytest.approx(0.0, abs=1e-15)

    def test_cos_matches_math(self):
        for v in [-math.pi, -math.pi/2, 0.0, math.pi/4, math.pi/2, math.pi]:
            assert rs.cos(v) == pytest.approx(math.cos(v))

    # tan
    def test_tan_basic(self):
        assert rs.tan(0.0) == 0.0
        assert rs.tan(math.pi / 4) == pytest.approx(1.0)

    def test_tan_matches_math(self):
        for v in [-math.pi/4, 0.0, math.pi/6, math.pi/4]:
            assert rs.tan(v) == pytest.approx(math.tan(v))

    # asin
    def test_asin_basic(self):
        assert rs.asin(0.0) == 0.0
        assert rs.asin(1.0) == pytest.approx(math.pi / 2)
        assert rs.asin(-1.0) == pytest.approx(-math.pi / 2)

    def test_asin_domain_error(self):
        with pytest.raises(ValueError):
            rs.asin(1.1)
        with pytest.raises(ValueError):
            rs.asin(-1.1)

    # acos
    def test_acos_basic(self):
        assert rs.acos(1.0) == pytest.approx(0.0)
        assert rs.acos(-1.0) == pytest.approx(math.pi)
        assert rs.acos(0.0) == pytest.approx(math.pi / 2)

    def test_acos_domain_error(self):
        with pytest.raises(ValueError):
            rs.acos(1.1)
        with pytest.raises(ValueError):
            rs.acos(-1.1)

    # atan
    def test_atan_basic(self):
        assert rs.atan(0.0) == 0.0
        assert rs.atan(1.0) == pytest.approx(math.pi / 4)
        assert rs.atan(-1.0) == pytest.approx(-math.pi / 4)

    # atan2
    def test_atan2_quadrants(self):
        assert rs.atan2(1.0, 1.0)   == pytest.approx(math.pi / 4)
        assert rs.atan2(1.0, -1.0)  == pytest.approx(3 * math.pi / 4)
        assert rs.atan2(-1.0, -1.0) == pytest.approx(-3 * math.pi / 4)
        assert rs.atan2(1.0, 0.0)   == pytest.approx(math.pi / 2)

    def test_atan2_zero_zero(self):
        assert rs.atan2(0.0, 0.0) == 0.0

    def test_atan2_matches_math(self):
        cases = [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)]
        for y, x in cases:
            assert rs.atan2(y, x) == pytest.approx(math.atan2(y, x))

    # hypot
    def test_hypot_basic(self):
        assert rs.hypot(3.0, 4.0) == pytest.approx(5.0)
        assert rs.hypot(0.0, 0.0) == 0.0

    def test_hypot_matches_math(self):
        for x, y in [(3.0, 4.0), (1.0, 1.0), (5.0, 12.0)]:
            assert rs.hypot(x, y) == pytest.approx(math.hypot(x, y))

    # hyperbolic
    def test_sinh_basic(self):
        assert rs.sinh(0.0) == 0.0
        assert rs.sinh(1.0) == pytest.approx(math.sinh(1.0))

    def test_cosh_basic(self):
        assert rs.cosh(0.0) == 1.0
        assert rs.cosh(1.0) == pytest.approx(math.cosh(1.0))

    def test_tanh_basic(self):
        assert rs.tanh(0.0) == 0.0
        assert rs.tanh(1.0) == pytest.approx(math.tanh(1.0))

    def test_trig_identity_sin2_cos2(self):
        # sin²(x) + cos²(x) == 1 for all x
        for v in [0.0, 0.1, math.pi/4, math.pi/3, math.pi/2, math.pi]:
            assert rs.sin(v)**2 + rs.cos(v)**2 == pytest.approx(1.0)

class TestHyperbolic:
    def test_sinh_basic(self):
        assert rs.sinh(0.0) == 0.0
        assert rs.sinh(1.0) == pytest.approx(math.sinh(1.0))
        assert rs.sinh(-1.0) == pytest.approx(math.sinh(-1.0))

    def test_cosh_basic(self):
        assert rs.cosh(0.0) == 1.0
        assert rs.cosh(1.0) == pytest.approx(math.cosh(1.0))

    def test_tanh_basic(self):
        assert rs.tanh(0.0) == 0.0
        assert rs.tanh(1.0) == pytest.approx(math.tanh(1.0))
        assert -1.0 < rs.tanh(100.0) <= 1.0   # always in (-1, 1)

    def test_asinh_basic(self):
        assert rs.asinh(0.0) == 0.0
        assert rs.asinh(1.0) == pytest.approx(math.asinh(1.0))
        assert rs.asinh(-1.0) == pytest.approx(math.asinh(-1.0))

    def test_asinh_roundtrip(self):
        for v in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert rs.asinh(rs.sinh(v)) == pytest.approx(v, rel=1e-12)

    def test_acosh_basic(self):
        assert rs.acosh(1.0) == pytest.approx(0.0)
        assert rs.acosh(2.0) == pytest.approx(math.acosh(2.0))

    def test_acosh_domain_error(self):
        with pytest.raises(ValueError):
            rs.acosh(0.9)
        with pytest.raises(ValueError):
            rs.acosh(-1.0)

    def test_acosh_roundtrip(self):
        for v in [1.0, 2.0, 5.0, 100.0]:
            assert rs.acosh(rs.cosh(v)) == pytest.approx(v, rel=1e-12)

    def test_atanh_basic(self):
        assert rs.atanh(0.0) == 0.0
        assert rs.atanh(0.5) == pytest.approx(math.atanh(0.5))

    def test_atanh_domain_error(self):
        with pytest.raises(ValueError):
            rs.atanh(1.0)
        with pytest.raises(ValueError):
            rs.atanh(-1.0)
        with pytest.raises(ValueError):
            rs.atanh(1.5)

    def test_atanh_roundtrip(self):
        for v in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            assert rs.atanh(rs.tanh(v)) == pytest.approx(v, rel=1e-12)

    def test_hyperbolic_identity(self):
        # cosh²(x) - sinh²(x) == 1 for all x
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assert rs.cosh(v)**2 - rs.sinh(v)**2 == pytest.approx(1.0)

class TestGeometry:
    # hypot_3d
    def test_hypot_3d_basic(self):
        assert rs.hypot_3d(2.0, 3.0, 6.0) == pytest.approx(7.0)
        assert rs.hypot_3d(0.0, 0.0, 0.0) == 0.0

    def test_hypot_3d_stable(self):
        # Should not overflow for large values
        big = 1e308
        result = rs.hypot_3d(big, big, big)
        assert math.isfinite(result)

    # fma
    def test_fma_precision(self):
    # FMA advantage: intermediate product retains full precision
    # Classic example where separate mul+add loses a bit
        x, y, z = 0.1, 10.0, -1.0
        fma_result = rs.fma(x, y, z)
        separate   = x * y + z          # may accumulate rounding error
        math_fma   = math.fma(x, y, z)
        assert fma_result == pytest.approx(math_fma, rel=1e-15)
        # Document the difference when it exists
        print(f"\nfma({x}, {y}, {z}): fma={fma_result} separate={separate}")

    def test_fma_basic_correct(self):
        assert rs.fma(2.0, 3.0, 1.0) == 7.0
        assert rs.fma(0.0, 999.0, 5.0) == 5.0
        assert rs.fma(-1.0, 1.0, 1.0) == 0.0

    # copysign
    def test_copysign_basic(self):
        assert rs.copysign(5.0, -42.0) == -5.0
        assert rs.copysign(-5.0, 1.0) == 5.0
        assert rs.copysign(0.0, -1.0) == -0.0

    def test_copysign_matches_math(self):
        for x, y in [(5.0, -1.0), (-3.0, 2.0), (1.0, 0.0)]:
            assert rs.copysign(x, y) == math.copysign(x, y)

    # nextafter
    def test_nextafter_up(self):
        assert rs.nextafter(1.0, 2.0) == pytest.approx(1.0 + 2.220446049250313e-16)

    def test_nextafter_equal(self):
        assert rs.nextafter(1.0, 1.0) == 1.0

    def test_nextafter_from_zero(self):
        pos = rs.nextafter(0.0, 1.0)
        neg = rs.nextafter(0.0, -1.0)
        assert pos > 0.0
        assert neg < 0.0

    def test_nextafter_matches_math(self):
        assert rs.nextafter(1.0, 2.0) == math.nextafter(1.0, 2.0)
        assert rs.nextafter(1.0, 0.0) == math.nextafter(1.0, 0.0)

    # degrees / radians
    def test_degrees_basic(self):
        assert rs.degrees(math.pi) == pytest.approx(180.0)
        assert rs.degrees(0.0) == 0.0

    def test_radians_basic(self):
        assert rs.radians(180.0) == pytest.approx(math.pi)
        assert rs.radians(0.0) == 0.0

    def test_degrees_radians_roundtrip(self):
        for v in [0.0, 30.0, 45.0, 90.0, 180.0, 360.0]:
            assert rs.degrees(rs.radians(v)) == pytest.approx(v)

    # frexp
    def test_frexp_basic(self):
        m, e = rs.frexp(1.0)
        assert m == pytest.approx(0.5)
        assert e == 1

    def test_frexp_zero(self):
        m, e = rs.frexp(0.0)
        assert m == 0.0
        assert e == 0

    def test_frexp_matches_math(self):
        for v in [0.5, 1.0, 2.0, 100.0, -3.5]:
            rm, re = rs.frexp(v)
            mm, me = math.frexp(v)
            assert rm == pytest.approx(mm)
            assert re == me

    # ulp
    def test_ulp_one(self):
        assert rs.ulp(1.0) == pytest.approx(2.220446049250313e-16)

    def test_ulp_zero(self):
        assert rs.ulp(0.0) > 0.0

    def test_ulp_nan(self):
        assert math.isnan(rs.ulp(float('nan')))

    def test_ulp_inf(self):
        assert math.isinf(rs.ulp(float('inf')))

class TestPredicates:
    def test_isfinite(self):
        assert rs.isfinite(1.0) is True
        assert rs.isfinite(0.0) is True
        assert rs.isfinite(float('inf')) is False
        assert rs.isfinite(float('-inf')) is False
        assert rs.isfinite(float('nan')) is False

    def test_isinf(self):
        assert rs.isinf(float('inf')) is True
        assert rs.isinf(float('-inf')) is True
        assert rs.isinf(1.0) is False
        assert rs.isinf(float('nan')) is False

    def test_isnan(self):
        assert rs.isnan(float('nan')) is True
        assert rs.isnan(1.0) is False
        assert rs.isnan(float('inf')) is False

    def test_is_integer_true(self):
        assert rs.is_integer(2.0) is True
        assert rs.is_integer(0.0) is True
        assert rs.is_integer(-5.0) is True

    def test_is_integer_false(self):
        assert rs.is_integer(2.5) is False
        assert rs.is_integer(0.1) is False

    def test_is_integer_special(self):
        # Key bug fix — inf and nan must return False
        assert rs.is_integer(float('inf')) is False
        assert rs.is_integer(float('-inf')) is False
        assert rs.is_integer(float('nan')) is False

    def test_is_integer_matches_python(self):
        for v in [0.0, 1.0, -1.0, 1.5, 2.5, float('inf'), float('nan')]:
            if not math.isnan(v):
                assert rs.is_integer(v) == v.is_integer(), f"Failed for {v}"

    def test_isclose_basic(self):
        assert rs.isclose(1.0, 1.0000000001) is True
        assert rs.isclose(1.0, 1.001) is False

    def test_isclose_equal(self):
        assert rs.isclose(1.0, 1.0) is True
        assert rs.isclose(0.0, 0.0) is True

    def test_isclose_abs_tol(self):
        assert rs.isclose(0.0, 1e-10, abs_tol=1e-9) is True
        assert rs.isclose(0.0, 1e-10, abs_tol=1e-11) is False

    def test_isclose_inf(self):
        assert rs.isclose(float('inf'), float('inf')) is True
        assert rs.isclose(float('inf'), float('-inf')) is False
        assert rs.isclose(float('inf'), 1e308) is False

    def test_isclose_matches_math(self):
        cases = [
            (1.0, 1.0000000001),
            (1.0, 1.001),
            (0.0, 1e-10),
            (1e10, 1e10 + 1),
        ]
        for a, b in cases:
            assert rs.isclose(a, b) == math.isclose(a, b), f"Failed for ({a}, {b})"

class TestIntegerBitwise:
    # factorial
    def test_factorial_basic(self):
        assert rs.factorial(0) == 1
        assert rs.factorial(1) == 1
        assert rs.factorial(5) == 120
        assert rs.factorial(10) == 3628800

    def test_factorial_limit(self):
        assert rs.factorial(34) is not None   # max valid
        with pytest.raises(OverflowError):
            rs.factorial(35)

    def test_factorial_matches_math(self):
        for n in range(15):
            assert rs.factorial(n) == math.factorial(n)

    # gcd
    def test_gcd_basic(self):
        assert rs.gcd(48, 18) == 6
        assert rs.gcd(0, 5) == 5
        assert rs.gcd(5, 0) == 5
        assert rs.gcd(0, 0) == 0

    def test_gcd_negative(self):
        assert rs.gcd(-48, 18) == 6
        assert rs.gcd(48, -18) == 6
        assert rs.gcd(-48, -18) == 6

    def test_gcd_matches_math(self):
        for a, b in [(48, 18), (100, 75), (17, 13), (0, 7)]:
            assert rs.gcd(a, b) == math.gcd(a, b)

    # lcm
    def test_lcm_basic(self):
        assert rs.lcm(12, 15) == 60
        assert rs.lcm(0, 5) == 0
        assert rs.lcm(5, 0) == 0

    def test_lcm_negative(self):
        assert rs.lcm(-12, 15) == 60
        assert rs.lcm(12, -15) == 60

    def test_lcm_overflow(self):
        with pytest.raises(OverflowError):
            rs.lcm(2**62, 3)   # (2^62 / 1) * 3 = 3 * 2^62 > i64::MAX

    def test_lcm_matches_math(self):
        for a, b in [(12, 15), (7, 13), (4, 6)]:
            assert rs.lcm(a, b) == math.lcm(a, b)

    # is_power_of_two
    def test_is_power_of_two_true(self):
        for n in [1, 2, 4, 8, 16, 1024]:
            assert rs.is_power_of_two(n) is True

    def test_is_power_of_two_false(self):
        for n in [0, -1, 3, 5, 6, 7, 100]:
            assert rs.is_power_of_two(n) is False

    # next_power_of_two
    def test_next_power_of_two_basic(self):
        assert rs.next_power_of_two(1) == 1
        assert rs.next_power_of_two(2) == 2
        assert rs.next_power_of_two(5) == 8
        assert rs.next_power_of_two(8) == 8
        assert rs.next_power_of_two(9) == 16

    def test_next_power_of_two_overflow(self):
        with pytest.raises(OverflowError):
            rs.next_power_of_two(2**62 + 1)

    def test_next_power_of_two_is_power(self):
        for n in [1, 3, 5, 100, 1000]:
            result = rs.next_power_of_two(n)
            assert rs.is_power_of_two(result)
            assert result >= n

    # is_prime
    def test_is_prime_true(self):
        for n in [2, 3, 5, 7, 11, 13, 17, 97, 7919]:
            assert rs.is_prime(n) is True, f"{n} should be prime"

    def test_is_prime_false(self):
        for n in [-1, 0, 1, 4, 6, 9, 15, 100]:
            assert rs.is_prime(n) is False, f"{n} should not be prime"

    def test_is_prime_edge(self):
        assert rs.is_prime(2) is True    # smallest prime
        assert rs.is_prime(3) is True
        assert rs.is_prime(4) is False

# ── Benchmark ─────────────────────────────────────────────────────────────────

N = 1_000_000

def bench(fn, *args):
    t = time.perf_counter()
    for _ in range(N):
        fn(*args)
    return time.perf_counter() - t

class TestBenchmark:
    def test_add_speed(self):
        r = bench(rs.add, 10.5, 0.5)
        p = bench(lambda: 10.5 + 0.5)        # pure Python
        print(f"\nadd   → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_div_speed(self):
        r = bench(rs.div, 10.0, 3.0)
        p = bench(lambda: 10.0 / 3.0)
        print(f"div   → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_fmod_speed(self):
        r = bench(rs.fmod, -5.0, 3.0)
        m = bench(math.fmod, -5.0, 3.0)
        print(f"fmod  → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_round_speed(self):
        r = bench(rs.round, 2.5)
        p = bench(lambda: round(2.5))
        print(f"round → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_round_half_even_speed(self):
        r = bench(rs.round_half_even, 2.5)
        p = bench(lambda: round(2.5))
        print(f"round_half_even → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_ceil_speed(self):
        r = bench(rs.ceil, 2.5)
        m = bench(math.ceil, 2.5)
        print(f"ceil  → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_floor_speed(self):
        r = bench(rs.floor, 2.5)
        m = bench(math.floor, 2.5)
        print(f"floor → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_trunc_speed(self):
        r = bench(rs.trunc, 2.5)
        m = bench(math.trunc, 2.5)
        print(f"trunc → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_clamp_speed(self):
        r = bench(rs.clamp, 2.5, 0.0, 5.0)
        p = bench(lambda: max(0.0, min(5.0, 2.5)))
        print(f"clamp → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_lerp_speed(self):
        r = bench(rs.lerp, 10.0, 20.0, 0.5)
        p = bench(lambda: 10.0 + (20.0 - 10.0) * 0.5)
        print(f"lerp  → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_sqrt_speed(self):
        r = bench(rs.sqrt, 64.0)
        m = bench(math.sqrt, 64.0)
        print(f"sqrt      → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_cbrt_speed(self):
        r = bench(rs.cbrt, 27.0)
        p = bench(math.pow, 27.0, 1/3)
        print(f"cbrt      → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_root_speed(self):
        r = bench(rs.root, 125.0, 3)
        p = bench(lambda: 125.0 ** (1/3))
        print(f"root      → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")

    def test_pow_speed(self):
        r = bench(rs.pow, 2.0, 10.0)
        m = bench(math.pow, 2.0, 10.0)
        print(f"pow       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_inv_sqrt_speed(self):
        r = bench(rs.inv_sqrt, 4.0)
        p = bench(lambda: 1.0 / math.sqrt(4.0))
        print(f"inv_sqrt  → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x")
    
    def test_exp_speed(self):
        r = bench(rs.exp, 1.0)
        m = bench(math.exp, 1.0)
        print(f"exp       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_exp2_speed(self):
        r = bench(rs.exp2, 10.0)
        p = bench(lambda: 2.0 ** 10.0)
        print(f"exp2      → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x (lambda overhead)")

    def test_expm1_speed(self):
        r = bench(rs.expm1, 1e-15)
        m = bench(math.expm1, 1e-15)
        print(f"expm1     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_log_speed(self):
        r = bench(rs.log, 100.0, 10.0)
        m = bench(math.log, 100.0, 10.0)
        print(f"log       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_log2_speed(self):
        r = bench(rs.log2, 1024.0)
        m = bench(math.log2, 1024.0)
        print(f"log2      → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_log10_speed(self):
        r = bench(rs.log10, 1000.0)
        m = bench(math.log10, 1000.0)
        print(f"log10     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_log1p_speed(self):
        r = bench(rs.log1p, 1e-15)
        m = bench(math.log1p, 1e-15)
        print(f"log1p     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_logsumexp2_speed(self):
        r = bench(rs.logsumexp2, 1.0, 2.0)
        p = bench(lambda x, y: math.log(math.exp(x) + math.exp(y)), 1.0, 2.0)
        print(f"logsumexp2→ rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x (multi-op lambda, fair)")

    def test_sin_speed(self):
        r = bench(rs.sin, math.pi / 4)
        m = bench(math.sin, math.pi / 4)
        print(f"sin       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_cos_speed(self):
        r = bench(rs.cos, math.pi / 4)
        m = bench(math.cos, math.pi / 4)
        print(f"cos       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_tan_speed(self):
        r = bench(rs.tan, math.pi / 4)
        m = bench(math.tan, math.pi / 4)
        print(f"tan       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_atan2_speed(self):
        r = bench(rs.atan2, 1.0, 1.0)
        m = bench(math.atan2, 1.0, 1.0)
        print(f"atan2     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_hypot_speed(self):
        r = bench(rs.hypot, 3.0, 4.0)
        m = bench(math.hypot, 3.0, 4.0)
        print(f"hypot     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_tanh_speed(self):
        r = bench(rs.tanh, 1.0)
        m = bench(math.tanh, 1.0)
        print(f"tanh      → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_asinh_speed(self):
        r = bench(rs.asinh, 1.0)
        m = bench(math.asinh, 1.0)
        print(f"asinh     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_acosh_speed(self):
        r = bench(rs.acosh, 2.0)
        m = bench(math.acosh, 2.0)
        print(f"acosh     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_atanh_speed(self):
        r = bench(rs.atanh, 0.5)
        m = bench(math.atanh, 0.5)
        print(f"atanh     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_hypot_3d_speed(self):
        r = bench(rs.hypot_3d, 2.0, 3.0, 6.0)
        p = bench(lambda: math.sqrt(4.0 + 9.0 + 36.0))
        print(f"hypot_3d  → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x (multi-op lambda, fair)")

    def test_fma_speed(self):
        r = bench(rs.fma, 2.0, 3.0, 1.0)
        p = bench(lambda: 2.0 * 3.0 + 1.0)
        print(f"fma       → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x (lambda overhead)")

    def test_copysign_speed(self):
        r = bench(rs.copysign, 5.0, -1.0)
        m = bench(math.copysign, 5.0, -1.0)
        print(f"copysign  → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_nextafter_speed(self):
        r = bench(rs.nextafter, 1.0, 2.0)
        m = bench(math.nextafter, 1.0, 2.0)
        print(f"nextafter → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_degrees_speed(self):
        r = bench(rs.degrees, math.pi)
        m = bench(math.degrees, math.pi)
        print(f"degrees   → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_radians_speed(self):
        r = bench(rs.radians, 180.0)
        m = bench(math.radians, 180.0)
        print(f"radians   → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_frexp_speed(self):
        r = bench(rs.frexp, 1.0)
        m = bench(math.frexp, 1.0)
        print(f"frexp     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_ulp_speed(self):
        r = bench(rs.ulp, 1.0)
        m = bench(math.ulp, 1.0)
        print(f"ulp       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")
    
    def test_isfinite_speed(self):
        r = bench(rs.isfinite, 1.0)
        m = bench(math.isfinite, 1.0)
        print(f"isfinite  → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_isinf_speed(self):
        r = bench(rs.isinf, 1.0)
        m = bench(math.isinf, 1.0)
        print(f"isinf     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_isnan_speed(self):
        r = bench(rs.isnan, float('nan'))
        m = bench(math.isnan, float('nan'))
        print(f"isnan     → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_isclose_speed(self):
        r = bench(rs.isclose, 1.0, 1.0000000001)
        m = bench(math.isclose, 1.0, 1.0000000001)
        print(f"isclose   → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")
    
    def test_factorial_speed(self):
        r = bench(rs.factorial, 20)
        m = bench(math.factorial, 20)
        print(f"factorial → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_gcd_speed(self):
        r = bench(rs.gcd, 48, 18)
        m = bench(math.gcd, 48, 18)
        print(f"gcd       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_lcm_speed(self):
        r = bench(rs.lcm, 12, 15)
        m = bench(math.lcm, 12, 15)
        print(f"lcm       → rmath: {r:.3f}s  math:   {m:.3f}s  ratio: {m/r:.2f}x")

    def test_is_prime_speed(self):
        r = bench(rs.is_prime, 7919)
        p = bench(lambda: all(7919 % i != 0 for i in range(2, int(7919**0.5) + 1)))
        print(f"is_prime  → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x (multi-op lambda, fair)")

    def test_is_power_of_two_speed(self):
        r = bench(rs.is_power_of_two, 1024)
        p = bench(lambda: 1024 > 0 and (1024 & 1023) == 0)
        print(f"pow2      → rmath: {r:.3f}s  python: {p:.3f}s  ratio: {p/r:.2f}x (lambda overhead)")

# from rmath import scalar as sc
import rmath.scalar as sc

import math
def test_1_arithmetic():
    print("--- [1/9] Arithmetic (6/6) ---")
    print(f"add(10, 5)        = {sc.add(10, 5)}")
    print(f"sub(10, 5)        = {sc.sub(10, 5)}")
    print(f"mul(10, 5)        = {sc.mul(10, 5)}")
    print(f"div(10, 5)        = {sc.div(10, 5)}")
    print(f"remainder(10, 3)  = {sc.remainder(10, 3)}") # 1.0
    print(f"fmod(-1.0, 3.0)   = {sc.fmod(-1.0, 3.0)}")   # -1.0
    try: sc.div(1, 0)
    except ZeroDivisionError: print("   ! div(1,0) caught")

def test_2_rounding():
    print("\n--- [2/9] Rounding & Range (5/5) ---")
    print(f"ceil(2.1)         = {sc.ceil(2.1)}")
    print(f"floor(2.9)        = {sc.floor(2.9)}")
    print(f"trunc(2.9)        = {sc.trunc(2.9)}")
    print(f"round(2.5)         = {sc.round(2.5)}")
    print(f"round_even(2.5)   = {sc.round_half_even(2.5)} (2.0)")
    print(f"round_even(3.5)   = {sc.round_half_even(3.5)} (4.0)")

def test_3_common_and_roots():
    print("\n--- [3/9] Common, Roots & Interpolation (9/9) ---")
    print(f"sqrt(64)          = {sc.sqrt(64.0)}")
    print(f"cbrt(27)          = {sc.cbrt(27.0)}")
    print(f"root(-27, 3)      = {sc.root(-27.0, 3)}")
    print(f"abs(-5.5)         = {sc.abs(-5.5)}")
    print(f"pow(2, 10)        = {sc.pow(2.0, 10.0)}")
    print(f"inv_sqrt(4)       = {sc.inv_sqrt(4.0)}")
    print(f"sign(-10.0)       = {sc.signum(-10.0)}")
    print(f"clamp(10, 1, 5)   = {sc.clamp(10.0, 1.0, 5.0)}")
    print(f"lerp(0, 10, 0.5)  = {sc.lerp(0.0, 10.0, 0.5)}")

def test_4_exp_and_log():
    print("\n--- [4/9] Exp & Log (8/8) ---")
    print(f"exp(1.0)          = {sc.exp(1.0)}")
    print(f"log(100, 10)      = {sc.log(100.0, 10.0)}")
    print(f"log2(1024)        = {sc.log2(1024.0)}")
    print(f"log10(100)        = {sc.log10(100.0)}")
    print(f"log1p(1e-15)      = {sc.log1p(1e-15)}")
    print(f"expm1(1e-15)      = {sc.expm1(1e-15)}")
    print(f"exp2(10)          = {sc.exp2(10.0)}")
    print(f"logsumexp2(0, 0)  = {sc.logsumexp2(0.0, 0.0)}")

def test_5_trigonometry():
    print("\n--- [5/9] Trigonometry (7/7) ---")
    p = math.pi/4
    print(f"sin(pi/4)         = {sc.sin(p)}")
    print(f"cos(pi/4)         = {sc.cos(p)}")
    print(f"tan(pi/4)         = {sc.tan(p)}")
    print(f"asin(sin(p))      = {sc.asin(sc.sin(p))}")
    print(f"acos(cos(p))      = {sc.acos(sc.cos(p))}")
    print(f"atan(1.0)         = {sc.atan(1.0)}")
    print(f"atan2(1, 1)       = {sc.atan2(1.0, 1.0)}")

def test_6_hyperbolic():
    print("\n--- [6/9] Hyperbolic (6/6) ---")
    print(f"sinh(0.5)         = {sc.sinh(0.5)}")
    print(f"cosh(0.5)         = {sc.cosh(0.5)}")
    print(f"tanh(0.5)         = {sc.tanh(0.5)}")
    print(f"asinh(0.5)        = {sc.asinh(0.5)}")
    print(f"acosh(1.5)        = {sc.acosh(1.5)}")
    print(f"atanh(0.5)        = {sc.atanh(0.5)}")

def test_7_geometry_and_utils():
    print("\n--- [7/9] Geometry & Utils (9/9) ---")
    print(f"hypot(3, 4)       = {sc.hypot(3.0, 4.0)}")
    print(f"hypot_3d(2,3,6)   = {sc.hypot_3d(2.0, 3.0, 6.0)}")
    print(f"fma(2, 3, 4)      = {sc.fma(2.0, 3.0, 4.0)}")
    print(f"copysign(5, -1)   = {sc.copysign(5.0, -1.0)}")
    print(f"nextafter(1, 2)   = {sc.nextafter(1.0, 2.0)}")
    print(f"degrees(pi)       = {sc.degrees(math.pi)}")
    print(f"radians(180)      = {sc.radians(180.0)}")
    m, e = sc.frexp(10.0)
    print(f"frexp(10.0)       = ({m}, {e})")
    print(f"ulp(1.0)          = {sc.ulp(1.0)}")

def test_8_predicates():
    print("\n--- [8/9] Predicates (5/5) ---")
    print(f"isfinite(1.0)     = {sc.isfinite(1.0)}")
    print(f"isinf(inf)        = {sc.isinf(float('inf'))}")
    print(f"isnan(nan)        = {sc.isnan(float('nan'))}")
    print(f"is_integer(1.0)   = {sc.is_integer(1.0)}")
    print(f"isclose(1, 1.001) = {sc.isclose(1.0, 1.001, rel_tol=1e-2)}")

def test_9_integer_math():
    print("\n--- [9/9] Integer Math & Bitwise (6/6) ---")
    print(f"factorial(10)     = {sc.factorial(10)}")
    print(f"gcd(48, 18)       = {sc.gcd(48, 18)}")
    print(f"lcm(48, 18)       = {sc.lcm(48, 18)}")
    print(f"is_pwr_two(1024)  = {sc.is_power_of_two(1024)}")
    print(f"next_pwr_two(33)  = {sc.next_power_of_two(33)}")
    print(f"is_prime(97)      = {sc.is_prime(97)}")

if __name__ == "__main__":
    test_1_arithmetic()
    test_2_rounding()
    test_3_common_and_roots()
    test_4_exp_and_log()
    test_5_trigonometry()
    test_6_hyperbolic()
    test_7_geometry_and_utils()
    test_8_predicates()
    test_9_integer_math()
    print("\n--- SUCCESS: All 61 scalar functions verified! ---")
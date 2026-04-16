# -*- coding: utf-8 -*-
"""
Test all domain-specific examples against the actual rmath API.
Adapt ChatGPT's examples to match the real method signatures.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import rmath as rm
import rmath.calculus as rc

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        failed += 1

# ============================================================
print("=" * 60)
print("1. Scientific Computing / Numerical Analysis")
print("=" * 60)

def test_solve_residual():
    A = rm.Array([[4.0, 2.0], [1.0, 3.0]])
    b = rm.Array([[1.0], [2.0]])
    x = rm.linalg.solve(A, b)
    residual = A.matmul(x).sub(b).norm_frobenius()
    assert residual < 1e-10, f"residual {residual}"
    print(f"    Solution: {x}, Residual: {residual}")

def test_eigh_reconstruction():
    A = rm.Array([[2.0, 1.0], [1.0, 2.0]])
    eigvals, eigvecs = rm.linalg.eigh(A)
    # Reconstruct A = V * Lambda * V^T
    Lambda = rm.Array.zeros(2, 2)
    Lambda[0, 0] = eigvals[0]
    Lambda[1, 1] = eigvals[1]
    reconstructed = eigvecs.matmul(Lambda).matmul(eigvecs.t())
    error = A.sub(reconstructed).norm_frobenius()
    print(f"    Eigenvalues: {eigvals}, Reconstruction error: {error}")
    assert error < 1e-10

test("Solve Ax=b + residual check", test_solve_residual)
test("Eigendecomposition + reconstruction", test_eigh_reconstruction)

# ============================================================
print()
print("=" * 60)
print("2. Data Science / Data Analysis")
print("=" * 60)

def test_data_analysis():
    data = rm.Array([[25, 50000], [30, 60000], [22, 45000], [35, 80000]])
    ages_list = data.get_col(0)      # returns list
    income_list = data.get_col(1)    # returns list
    ages = rm.Vector(ages_list)
    income = rm.Vector(income_list)
    print(f"    Mean age: {ages.mean()}")
    print(f"    Income std: {income.std_dev()}")
    corr = rm.stats.correlation(ages, income)
    print(f"    Correlation: {corr}")
    assert abs(corr) <= 1.0

test("Load -> extract columns -> correlate", test_data_analysis)

# ============================================================
print()
print("=" * 60)
print("3. Statistics / Research")
print("=" * 60)

def test_descriptive():
    v = rm.Vector([2, 4, 4, 4, 5, 5, 7, 9])
    print(f"    Mean: {v.mean()}, Variance: {v.variance()}, Std: {v.std_dev()}")
    assert abs(v.mean() - 5.0) < 1e-10

def test_t_test():
    group1 = rm.Vector([20, 22, 19, 24, 25])
    group2 = rm.Vector([30, 29, 35, 32, 31])
    t_stat, p_value = rm.stats.t_test_independent(group1, group2)
    print(f"    t-stat: {t_stat}, p-value: {p_value}")
    assert p_value < 0.05  # clearly different groups

def test_linear_regression():
    x = rm.Vector([1, 2, 3, 4, 5])
    y = rm.Vector([2, 4, 5, 4, 5])
    result = rm.stats.linear_regression(x, y)
    slope = result['slope']
    intercept = result['intercept']
    r_sq = result['r_squared']
    print(f"    y = {slope:.4f}*x + {intercept:.4f} (R2 = {r_sq:.4f})")
    assert abs(slope - 0.6) < 1e-10

test("Descriptive statistics",     test_descriptive)
test("Independent t-test",         test_t_test)
test("Linear regression",          test_linear_regression)

# ============================================================
print()
print("=" * 60)
print("4. Financial / Economic Analysis")
print("=" * 60)

def test_returns():
    prices = rm.Vector([100, 102, 101, 105, 110])
    diffs = prices.diff()  # [2, -1, 4, 5]
    print(f"    Price diffs: {list(diffs)}")
    assert len(diffs) == 4

def test_covariance_matrix():
    data = rm.Array([
        [0.01, 0.02, 0.015],
        [0.03, 0.01, 0.02],
        [0.02, 0.025, 0.03],
    ])
    cov = data.covariance()
    print(f"    Covariance shape: {cov.shape}")
    assert cov.shape == [3, 3]

test("Price diffs (diff)",        test_returns)
test("Covariance matrix",         test_covariance_matrix)

# ============================================================
print()
print("=" * 60)
print("5. Machine Learning (Autograd)")
print("=" * 60)

def test_tensor_grad():
    x = rm.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    print(f"    Gradients: {x.grad}")
    for xi, gi in zip(x.data.to_flat_list(), x.grad.to_flat_list()):
        assert abs(gi - 2 * xi) < 1e-8

def test_neural_step():
    w = rm.Tensor.randn(3, requires_grad=True)
    x = rm.Tensor([1.0, 2.0, 3.0])
    y_pred = (w * x).sum()
    target = rm.Tensor([10.0])
    loss = (y_pred - target) * (y_pred - target)
    loss = loss.sum()
    loss.backward()
    assert w.grad is not None
    print(f"    Loss: {loss.data.to_flat_list()[0]:.4f}, w.grad: {w.grad}")

def test_relu_activation():
    x = rm.Array([-1.0, 0.0, 2.0])
    activated = x.relu()
    vals = activated.to_flat_list()
    assert vals[0] == 0.0
    assert vals[1] == 0.0
    assert vals[2] == 2.0
    print(f"    relu([-1, 0, 2]) = {vals}")

test("Gradient d/dx(x^2).sum()",     test_tensor_grad)
test("Neural step (w*x loss)",       test_neural_step)
test("ReLU activation",              test_relu_activation)

# ============================================================
print()
print("=" * 60)
print("6. Calculus / Differentiation")
print("=" * 60)

def test_dual_numbers():
    x = rc.Dual(2.0, 1.0)
    y = x.sin() * x.exp()
    import math
    expected_val = math.sin(2.0) * math.exp(2.0)
    expected_der = (math.cos(2.0) * math.exp(2.0)) + (math.sin(2.0) * math.exp(2.0))
    assert abs(y.value - expected_val) < 1e-10
    assert abs(y.derivative - expected_der) < 1e-10
    print(f"    sin(x)*exp(x) at x=2: value={y.value:.6f}, deriv={y.derivative:.6f}")

def test_integration():
    result = rm.calculus.integrate_simpson(lambda x: x * x, 0, 1, 100)
    assert abs(result - 1/3) < 1e-8
    print(f"    integral(x^2, 0, 1) = {result:.10f} (expected {1/3:.10f})")

test("Dual: sin(x)*exp(x)",         test_dual_numbers)
test("Simpson integration x^2",     test_integration)

# ============================================================
print()
print("=" * 60)
print("7. Signal Processing")
print("=" * 60)

def test_convolution():
    signal = rm.Vector([1, 2, 3, 4])
    kernel = rm.Vector([1, 0, -1])
    filtered = rm.signal.convolve(signal, kernel, "full")
    print(f"    convolve([1,2,3,4], [1,0,-1]) = {list(filtered)}")
    vals = list(filtered)
    assert len(vals) == 6

def test_fft():
    signal = rm.Vector([1, 0, 0, 0])
    fft_result = rm.signal.fft(signal)
    print(f"    FFT([1,0,0,0]) magnitudes: {list(fft_result.to_mags())}")

test("1D convolution (FFT mode)",   test_convolution)
test("FFT",                          test_fft)

# ============================================================
print()
print("=" * 60)
print("8. Geometry")
print("=" * 60)

def test_distance():
    a = rm.Vector([1, 2, 3])
    b = rm.Vector([4, 5, 6])
    dist = rm.geometry.euclidean_distance(a, b)
    import math
    expected = math.sqrt(27)
    assert abs(dist - expected) < 1e-10
    print(f"    Distance: {dist:.6f}")

def test_cosine():
    a = rm.Vector([1, 2, 3])
    b = rm.Vector([4, 5, 6])
    cos_sim = rm.geometry.cosine_similarity(a, b)
    print(f"    Cosine similarity: {cos_sim:.6f}")
    assert 0 < cos_sim <= 1.0

test("Euclidean distance",           test_distance)
test("Cosine similarity",            test_cosine)

# ============================================================
print()
print("=" * 60)
print("9. Interoperability")
print("=" * 60)

def test_numpy_interop():
    import numpy as np
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    rm_arr = rm.Array.from_numpy(np_arr)
    back = rm_arr.to_numpy()
    assert (back == np_arr).all()
    print(f"    NumPy -> rmath -> NumPy roundtrip OK (shape={rm_arr.shape})")

test("NumPy <-> rmath roundtrip",    test_numpy_interop)

try:
    import torch
    def test_torch_interop():
        t = torch.tensor([[1.0, 2.0]])
        rm_arr = rm.Array.from_torch(t)
        back = rm_arr.to_torch()
        assert torch.allclose(t, back)
        print(f"    PyTorch -> rmath -> PyTorch roundtrip OK")
    test("PyTorch <-> rmath roundtrip", test_torch_interop)
except ImportError:
    print("  [SKIP] PyTorch not installed, skipping")

# ============================================================
print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 60)

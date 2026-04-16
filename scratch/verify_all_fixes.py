"""Verify all fixes: module paths, Tensor requires_grad, new Tensor methods, README examples."""
import rmath as rm
import rmath.vector as rv
import rmath.calculus as rc

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    print(f"  {name}...", end=" ")
    try:
        fn()
        print("PASS")
        passed += 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        failed += 1

print("=" * 60)
print("FIX 1: Module paths (builtins -> rmath)")
print("=" * 60)

def check_module(cls, expected):
    actual = f"{cls.__module__}.{cls.__qualname__}"
    assert actual == expected, f"got {actual}"

test("Vector",   lambda: check_module(rm.Vector,   "rmath.Vector"))
test("Array",    lambda: check_module(rm.Array,     "rmath.Array"))
test("Scalar",   lambda: check_module(rm.Scalar,    "rmath.Scalar"))
test("Tensor",   lambda: check_module(rm.Tensor,    "rmath.Tensor"))
test("Dual",     lambda: check_module(rm.Dual,      "rmath.Dual"))
test("LazyArray", lambda: check_module(rm.LazyArray, "rmath.LazyArray"))

print()
print("=" * 60)
print("FIX 2: Tensor.randn(requires_grad=True) now works")
print("=" * 60)

def test_randn_grad():
    x = rm.Tensor.randn(3, requires_grad=True)
    assert x.requires_grad == True, f"got {x.requires_grad}"

def test_zeros_grad():
    x = rm.Tensor.zeros(3, requires_grad=True)
    assert x.requires_grad == True, f"got {x.requires_grad}"

def test_ones_grad():
    x = rm.Tensor.ones(3, requires_grad=True)
    assert x.requires_grad == True, f"got {x.requires_grad}"

test("randn(requires_grad=True)", test_randn_grad)
test("zeros(requires_grad=True)", test_zeros_grad)
test("ones(requires_grad=True)",  test_ones_grad)

print()
print("=" * 60)
print("FIX 3: New Tensor methods + autograd")
print("=" * 60)

def test_tensor_sum():
    x = rm.Tensor.randn(5, requires_grad=True)
    y = (x * x).sum()
    y.backward()
    assert x.grad is not None, "grad is None"
    print(f"(grad={x.grad})", end=" ")

def test_tensor_mean():
    x = rm.Tensor.ones(4, requires_grad=True)
    y = x.mean()
    assert abs(y.data.to_flat_list()[0] - 1.0) < 1e-10

def test_tensor_exp():
    x = rm.Tensor.zeros(3, requires_grad=True)
    y = x.exp().sum()
    y.backward()
    # d/dx exp(0) = 1.0
    for v in x.grad.to_flat_list():
        assert abs(v - 1.0) < 1e-10, f"expected 1.0, got {v}"

def test_tensor_log():
    x = rm.Tensor.ones(3, requires_grad=True)
    y = x.log().sum()
    y.backward()
    # d/dx log(1) = 1.0
    for v in x.grad.to_flat_list():
        assert abs(v - 1.0) < 1e-10, f"expected 1.0, got {v}"

def test_tensor_tanh():
    x = rm.Tensor.zeros(3, requires_grad=True)
    y = x.tanh().sum()
    y.backward()
    # d/dx tanh(0) = 1 - 0^2 = 1.0
    for v in x.grad.to_flat_list():
        assert abs(v - 1.0) < 1e-10, f"expected 1.0, got {v}"

def test_tensor_neg():
    x = rm.Tensor.ones(3, requires_grad=True)
    y = (-x).sum()
    y.backward()
    # d/dx (-x) = -1
    for v in x.grad.to_flat_list():
        assert abs(v - (-1.0)) < 1e-10, f"expected -1.0, got {v}"

def test_tensor_abs():
    x = rm.Tensor([-2.0, 3.0, -1.0], requires_grad=True)
    y = x.abs().sum()
    y.backward()
    expected = [-1.0, 1.0, -1.0]  # sign(x)
    for v, e in zip(x.grad.to_flat_list(), expected):
        assert abs(v - e) < 1e-10, f"expected {e}, got {v}"

def test_tensor_reshape():
    x = rm.Tensor.randn(2, 3)
    y = x.reshape([3, 2])
    assert y.shape == [3, 2]

def test_tensor_transpose():
    x = rm.Tensor.randn(2, 3)
    y = x.t()
    assert y.shape == [3, 2]

test("sum() + backward",     test_tensor_sum)
test("mean()",               test_tensor_mean)
test("exp() + backward",     test_tensor_exp)
test("log() + backward",     test_tensor_log)
test("tanh() + backward",    test_tensor_tanh)
test("neg() + backward",     test_tensor_neg)
test("abs() + backward",     test_tensor_abs)
test("reshape()",            test_tensor_reshape)
test("transpose() / t()",    test_tensor_transpose)

print()
print("=" * 60)
print("README examples (all must pass)")
print("=" * 60)

def test_readme_hero():
    data = rm.Array.randn(100, 100)
    avg = data.mean()
    std = data.std_dev()
    b = rm.Array.ones(100, 1)
    x = rm.linalg.solve(data, b)
    assert x.shape == [100, 1]

def test_readme_vector():
    v = rv.Vector.linspace(0, 10, 100_000)
    result = v.sin().exp().sum()
    assert result > 0

def test_readme_stats():
    v = rm.Vector.randn(1000)
    report = rm.stats.describe(v)
    assert "mean" in report

def test_readme_dual():
    val = rc.Dual(2.0, 1.0)
    out = val * val + val * 3.0
    assert abs(out.value - 10.0) < 1e-10
    assert abs(out.derivative - 7.0) < 1e-10

def test_readme_tensor_autodiff():
    x = rm.Tensor.randn(3, requires_grad=True)
    y = (x * x).sum()
    y.backward()
    assert x.grad is not None
    # grad should be 2*x
    for xi, gi in zip(x.data.to_flat_list(), x.grad.to_flat_list()):
        assert abs(gi - 2.0 * xi) < 1e-8, f"expected {2*xi}, got {gi}"

test("Hero example (mean + std_dev + solve)", test_readme_hero)
test("Vector chain (sin.exp.sum)",            test_readme_vector)
test("stats.describe(Vector)",                test_readme_stats)
test("Dual arithmetic (x^2+3x)",             test_readme_dual)
test("Tensor autodiff (x*x).sum().backward()", test_readme_tensor_autodiff)

print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

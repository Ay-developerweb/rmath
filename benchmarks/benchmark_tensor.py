"""
Tensor (Autograd) Benchmark Suite
=================================
Tests forward pass, backward pass, and training loop performance.
Compares against PyTorch where available.

Run:  python -m benchmarks.benchmark_tensor
"""

import time
import sys

def timer(fn, warmup=3, runs=20):
    """Time a function, return median in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]

# ── Import rmath ──────────────────────────────────────────────────────────
from rmath import Tensor, Array

# ── Try importing PyTorch ─────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

WIDTH = 90
passed = 0
failed = 0
results = []

def section(title):
    print(f"\n── {title} {'─' * (WIDTH - len(title) - 4)}")

def bench(label, rm_fn, torch_fn=None, check_fn=None):
    global passed, failed
    rm_t = timer(rm_fn)
    rm_us = rm_t * 1e6

    if torch_fn and HAS_TORCH:
        pt_t = timer(torch_fn)
        pt_us = pt_t * 1e6
        ratio = pt_t / rm_t
        line = f"  BENCH   {label:<42} rm={rm_us:>8.1f} µs  pt={pt_us:>8.1f} µs  vPT={ratio:>5.1f}x"
    else:
        pt_us = None
        ratio = None
        line = f"  BENCH   {label:<42} rm={rm_us:>8.1f} µs"
    
    results.append((label, rm_us, pt_us, ratio))
    print(line)

    if check_fn:
        try:
            check_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {label}: {e}")
    else:
        passed += 1

def check(label, check_fn):
    global passed, failed
    try:
        check_fn()
        passed += 1
        print(f"  [PASS] {label}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] {label}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

section("1. Construction")

def check_from_list():
    t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert t.shape == [3]
    assert t.requires_grad == True

check("Tensor from list", check_from_list)

def check_zeros():
    t = Tensor.zeros(5, 5, requires_grad=True)
    assert t.shape == [5, 5]

check("Tensor.zeros", check_zeros)

def check_ones():
    t = Tensor.ones(3, 4)
    assert t.shape == [3, 4]

check("Tensor.ones", check_ones)

def check_randn():
    t = Tensor.randn(10, 10, requires_grad=True)
    assert t.shape == [10, 10]

check("Tensor.randn", check_randn)

# ── Forward pass ──────────────────────────────────────────────────────────

section("2. Forward Pass (no grad)")

SIZE = 200

def rm_add():
    a = Tensor.randn(SIZE, SIZE)
    b = Tensor.randn(SIZE, SIZE)
    return a + b

def pt_add():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return a + b

bench(f"add {SIZE}×{SIZE}", rm_add, pt_add if HAS_TORCH else None)

def rm_mul():
    a = Tensor.randn(SIZE, SIZE)
    b = Tensor.randn(SIZE, SIZE)
    return a * b

def pt_mul():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return a * b

bench(f"mul {SIZE}×{SIZE}", rm_mul, pt_mul if HAS_TORCH else None)

def rm_matmul():
    a = Tensor.randn(SIZE, SIZE)
    b = Tensor.randn(SIZE, SIZE)
    return a @ b

def pt_matmul():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return a @ b

bench(f"matmul {SIZE}×{SIZE}", rm_matmul, pt_matmul if HAS_TORCH else None)

def rm_sigmoid():
    a = Tensor.randn(SIZE, SIZE)
    return a.sigmoid()

def pt_sigmoid():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return torch.sigmoid(a)

bench(f"sigmoid {SIZE}×{SIZE}", rm_sigmoid, pt_sigmoid if HAS_TORCH else None)

def rm_tanh():
    a = Tensor.randn(SIZE, SIZE)
    return a.tanh()

def pt_tanh():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return torch.tanh(a)

bench(f"tanh {SIZE}×{SIZE}", rm_tanh, pt_tanh if HAS_TORCH else None)

def rm_exp():
    a = Tensor.randn(SIZE, SIZE)
    return a.exp()

def pt_exp():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return torch.exp(a)

bench(f"exp {SIZE}×{SIZE}", rm_exp, pt_exp if HAS_TORCH else None)

# ── Forward pass with grad ────────────────────────────────────────────────

section("3. Forward Pass (with grad tracking)")

def rm_add_grad():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    b = Tensor.randn(SIZE, SIZE, requires_grad=True)
    return a + b

def pt_add_grad():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    return a + b

bench(f"add (grad) {SIZE}×{SIZE}", rm_add_grad, pt_add_grad if HAS_TORCH else None)

def rm_chain_grad():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    b = Tensor.randn(SIZE, SIZE, requires_grad=True)
    c = (a * b).sigmoid()
    return c.sum()

def pt_chain_grad():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    c = (a * b).sigmoid()
    return c.sum()

bench(f"mul→sigmoid→sum (grad) {SIZE}×{SIZE}", rm_chain_grad, pt_chain_grad if HAS_TORCH else None)

# ── Backward pass ─────────────────────────────────────────────────────────

section("4. Backward Pass")

def rm_backward_add():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    b = Tensor.randn(SIZE, SIZE, requires_grad=True)
    c = (a + b).sum()
    c.backward()

def pt_backward_add():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    c = (a + b).sum()
    c.backward()

bench(f"add→sum→backward {SIZE}×{SIZE}", rm_backward_add, pt_backward_add if HAS_TORCH else None)

def rm_backward_mul():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    b = Tensor.randn(SIZE, SIZE, requires_grad=True)
    c = (a * b).sum()
    c.backward()

def pt_backward_mul():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    c = (a * b).sum()
    c.backward()

bench(f"mul→sum→backward {SIZE}×{SIZE}", rm_backward_mul, pt_backward_mul if HAS_TORCH else None)

def rm_backward_sigmoid():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    c = a.sigmoid().sum()
    c.backward()

def pt_backward_sigmoid():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    c = torch.sigmoid(a).sum()
    c.backward()

bench(f"sigmoid→sum→backward {SIZE}×{SIZE}", rm_backward_sigmoid, pt_backward_sigmoid if HAS_TORCH else None)

def rm_backward_tanh():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    c = a.tanh().sum()
    c.backward()

def pt_backward_tanh():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    c = torch.tanh(a).sum()
    c.backward()

bench(f"tanh→sum→backward {SIZE}×{SIZE}", rm_backward_tanh, pt_backward_tanh if HAS_TORCH else None)

def rm_backward_matmul():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    b = Tensor.randn(SIZE, SIZE, requires_grad=True)
    c = (a @ b).sum()
    c.backward()

def pt_backward_matmul():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    b = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    c = (a @ b).sum()
    c.backward()

bench(f"matmul→sum→backward {SIZE}×{SIZE}", rm_backward_matmul, pt_backward_matmul if HAS_TORCH else None)

def rm_backward_div():
    a = Tensor.randn(SIZE, SIZE, requires_grad=True)
    b = Tensor.ones(SIZE, SIZE, requires_grad=True) * 2.0
    c = (a / b).sum()
    c.backward()

def pt_backward_div():
    a = torch.randn(SIZE, SIZE, dtype=torch.float64, requires_grad=True)
    b = torch.full((SIZE, SIZE), 2.0, dtype=torch.float64, requires_grad=True)
    c = (a / b).sum()
    c.backward()

bench(f"div→sum→backward {SIZE}×{SIZE}", rm_backward_div, pt_backward_div if HAS_TORCH else None)

# ── Correctness checks ───────────────────────────────────────────────────

section("5. Gradient Correctness")

def check_add_grad():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = (a + b).sum()
    c.backward()
    ag = a.grad.to_flat_list()
    bg = b.grad.to_flat_list()
    assert all(abs(g - 1.0) < 1e-10 for g in ag), f"add grad_a wrong: {ag}"
    assert all(abs(g - 1.0) < 1e-10 for g in bg), f"add grad_b wrong: {bg}"

check("add gradients = [1,1,1]", check_add_grad)

def check_mul_grad():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = (a * b).sum()
    c.backward()
    ag = a.grad.to_flat_list()
    bg = b.grad.to_flat_list()
    # d/da (a*b) = b, d/db (a*b) = a
    assert all(abs(ag[i] - [4,5,6][i]) < 1e-10 for i in range(3)), f"mul grad_a wrong: {ag}"
    assert all(abs(bg[i] - [1,2,3][i]) < 1e-10 for i in range(3)), f"mul grad_b wrong: {bg}"

check("mul gradients (a*b): da=b, db=a", check_mul_grad)

def check_sigmoid_grad():
    a = Tensor([0.0], requires_grad=True)
    c = a.sigmoid().sum()
    c.backward()
    # sigmoid(0) = 0.5, sigmoid'(0) = 0.25
    g = a.grad.to_flat_list()[0]
    assert abs(g - 0.25) < 1e-10, f"sigmoid grad wrong: {g}"

check("sigmoid grad at 0 = 0.25", check_sigmoid_grad)

def check_tanh_grad():
    a = Tensor([0.0], requires_grad=True)
    c = a.tanh().sum()
    c.backward()
    # tanh(0) = 0, tanh'(0) = 1
    g = a.grad.to_flat_list()[0]
    assert abs(g - 1.0) < 1e-10, f"tanh grad wrong: {g}"

check("tanh grad at 0 = 1.0", check_tanh_grad)

def check_exp_grad():
    a = Tensor([0.0], requires_grad=True)
    c = a.exp().sum()
    c.backward()
    # exp(0) = 1, exp'(0) = 1
    g = a.grad.to_flat_list()[0]
    assert abs(g - 1.0) < 1e-10, f"exp grad wrong: {g}"

check("exp grad at 0 = 1.0", check_exp_grad)

def check_div_grad():
    a = Tensor([6.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = (a / b).sum()
    c.backward()
    # d/da (a/b) = 1/b = 1/3
    # d/db (a/b) = -a/b² = -6/9 = -2/3
    ga = a.grad.to_flat_list()[0]
    gb = b.grad.to_flat_list()[0]
    assert abs(ga - 1/3) < 1e-10, f"div grad_a wrong: {ga}"
    assert abs(gb - (-2/3)) < 1e-10, f"div grad_b wrong: {gb}"

check("div grad: da=1/b, db=-a/b²", check_div_grad)

def check_neg_grad():
    a = Tensor([1.0, 2.0], requires_grad=True)
    c = (-a).sum()
    c.backward()
    g = a.grad.to_flat_list()
    assert all(abs(gi - (-1.0)) < 1e-10 for gi in g), f"neg grad wrong: {g}"

check("neg grad = [-1, -1]", check_neg_grad)

def check_chain():
    # f(x) = sigmoid(x²).sum()
    # f'(x) = sigmoid(x²) * (1 - sigmoid(x²)) * 2x
    x = Tensor([1.0], requires_grad=True)
    y = (x * x).sigmoid().sum()
    y.backward()
    import math
    sx2 = 1.0 / (1.0 + math.exp(-1.0))
    expected = sx2 * (1 - sx2) * 2.0
    g = x.grad.to_flat_list()[0]
    assert abs(g - expected) < 1e-8, f"chain grad wrong: {g} vs {expected}"

check("chain rule: sigmoid(x²)", check_chain)

# ── Training loop simulation ─────────────────────────────────────────────

section("6. Training Loop Simulation")

def rm_training_step():
    # Simple: y = Wx, loss = (y - target)², backward, update
    W = Tensor.randn(50, 50, requires_grad=True)
    x = Tensor.randn(50, 1)
    target = Tensor.randn(50, 1)
    
    y = W @ x
    diff = y - target
    loss = (diff * diff).sum()
    loss.backward()

def pt_training_step():
    W = torch.randn(50, 50, dtype=torch.float64, requires_grad=True)
    x = torch.randn(50, 1, dtype=torch.float64)
    target = torch.randn(50, 1, dtype=torch.float64)
    
    y = W @ x
    diff = y - target
    loss = (diff * diff).sum()
    loss.backward()

bench("training step 50×50", rm_training_step, pt_training_step if HAS_TORCH else None)

def rm_training_100():
    W = Tensor.randn(100, 100, requires_grad=True)
    x = Tensor.randn(100, 1)
    target = Tensor.randn(100, 1)
    y = W @ x
    diff = y - target
    loss = (diff * diff).sum()
    loss.backward()

def pt_training_100():
    W = torch.randn(100, 100, dtype=torch.float64, requires_grad=True)
    x = torch.randn(100, 1, dtype=torch.float64)
    target = torch.randn(100, 1, dtype=torch.float64)
    y = W @ x
    diff = y - target
    loss = (diff * diff).sum()
    loss.backward()

bench("training step 100×100", rm_training_100, pt_training_100 if HAS_TORCH else None)

# ── Shape ops ─────────────────────────────────────────────────────────────

section("7. Shape Operations")

def rm_reshape():
    t = Tensor.randn(SIZE, SIZE)
    return t.reshape([SIZE * SIZE])

def pt_reshape():
    t = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return t.reshape(SIZE * SIZE)

bench(f"reshape {SIZE}×{SIZE}→flat", rm_reshape, pt_reshape if HAS_TORCH else None)

def rm_transpose():
    t = Tensor.randn(SIZE, SIZE)
    return t.transpose()

def pt_transpose():
    t = torch.randn(SIZE, SIZE, dtype=torch.float64)
    return t.T.contiguous()

bench(f"transpose {SIZE}×{SIZE}", rm_transpose, pt_transpose if HAS_TORCH else None)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * WIDTH}")
print("SUMMARY")
print(f"{'═' * WIDTH}")
print(f"  Tests:   {passed + failed}")
print(f"  Passed:  {passed}  ({100 * passed / max(1, passed + failed):.0f}%)")
print(f"  Failed:  {failed}")

if results:
    has_pt = any(r[3] is not None for r in results)
    if has_pt:
        pt_results = [(r[0], r[3]) for r in results if r[3] is not None]
        pt_results.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 speedups vs PyTorch:")
        for label, ratio in pt_results[:5]:
            marker = "🔥" if ratio > 1 else "  "
            print(f"    {marker} {ratio:>6.2f}×  {label}")
        
        avg = sum(r[1] for r in pt_results) / len(pt_results)
        print(f"\n  Average speedup vs PyTorch: {avg:.2f}×")

if failed:
    sys.exit(1)

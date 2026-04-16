"""Test all code examples from ChatGPT's rmath presentation review."""
import rmath as rm
import rmath.vector as rv
import rmath.calculus as rc

def test(name, fn):
    print(f"=== {name} ===")
    try:
        fn()
        print("  ✓ PASS\n")
    except Exception as e:
        print(f"  ✗ FAIL: {type(e).__name__}: {e}\n")

# --- ChatGPT's Example 1: Vector + Stats ---
def t1():
    v = rm.Vector.arange(1, 1000000)
    print(f"  mean={v.mean()}, std_dev={v.std_dev()}")
test("ChatGPT #1: Vector.arange + mean/std_dev", t1)

# --- ChatGPT's Example 2: Array.randn + inv ---
def t2():
    A = rm.Array.randn(100, 100)
    inv_A = A.inv()
    print(f"  inv shape={inv_A.shape}")
test("ChatGPT #2: Array.randn(100,100).inv()", t2)

# --- ChatGPT's Example 3: Tensor autodiff ---
def t3():
    x = rm.Tensor.randn(3, requires_grad=True)
    y = (x * x)
    # ChatGPT said y.sum().backward() — check if Tensor has .sum()
    methods = [m for m in dir(y) if not m.startswith("_")]
    print(f"  Tensor public methods: {methods}")
    # Try backward directly on y (element-wise)
    y.backward()
    print(f"  x.grad = {x.grad}")
test("ChatGPT #3: Tensor autodiff (x*x).backward()", t3)

# --- ChatGPT's Example 4: Dual numbers ---
def t4():
    x = rm.Dual(2.0, 1.0)
    y = x.sin() * x.exp()
    print(f"  value={y.value}, derivative={y.derivative}")
test("ChatGPT #4: Dual sin()*exp()", t4)

# --- README Example: Array.randn + mean + std ---
def t5():
    data = rm.Array.randn(1000, 1000)
    avg = data.mean()
    print(f"  mean={avg}")
    # Check if std() works or if it's std_dev()
    try:
        std = data.std()
        print(f"  std()={std}")
    except AttributeError:
        std = data.std_dev()
        print(f"  std_dev()={std} (note: .std() doesn't exist, use .std_dev())")
test("README Example: data.mean(), data.std()", t5)

# --- README Example: linalg.solve ---
def t6():
    data = rm.Array.randn(1000, 1000)
    b = rm.Array.ones(1000, 1)
    x = rm.linalg.solve(data, b)
    print(f"  solution shape={x.shape}")
test("README Example: linalg.solve()", t6)

# --- README Example: Vector.linspace chain ---
def t7():
    v = rv.Vector.linspace(0, 10, 1_000_000)
    result = v.sin().exp().sum()
    print(f"  sin().exp().sum() = {result}")
test("README Example: Vector.linspace chain", t7)

# --- README Example: stats.describe ---
def t8():
    data = rm.Array.randn(10_000, 1)
    report = rm.stats.describe(data)
    print(f"  describe = {report}")
test("README Example: stats.describe()", t8)

# --- README Example: Dual arithmetic ---
def t9():
    val = rc.Dual(2.0, 1.0)
    out = val * val + val * 3.0
    print(f"  value={out.value} (expect 10.0)")
    print(f"  derivative={out.derivative} (expect 7.0)")
test("README Example: Dual arithmetic f(x)=x²+3x", t9)

# --- Check: does Tensor have .sum()? ---
def t10():
    x = rm.Tensor.randn(3, requires_grad=True)
    methods = sorted([m for m in dir(x) if not m.startswith("_")])
    print(f"  Tensor methods: {methods}")
    has_sum = hasattr(x, "sum")
    print(f"  has .sum()? {has_sum}")
test("API Check: Tensor capabilities", t10)

# --- Check: Array.std() vs Array.std_dev() ---
def t11():
    a = rm.Array.randn(10, 10)
    methods = sorted([m for m in dir(a) if "std" in m.lower()])
    print(f"  std-related methods: {methods}")
test("API Check: Array std methods", t11)

# --- Check: Tensor.randn signature ---
def t12():
    # ChatGPT used: rm.Tensor.randn(3, requires_grad=True)
    # Check actual signature
    try:
        x = rm.Tensor.randn(3, requires_grad=True)
        print(f"  randn(3, requires_grad=True) => shape={x.shape}")
    except TypeError as e:
        print(f"  randn(3, requires_grad=True) failed: {e}")
        try:
            x = rm.Tensor.randn(3)
            x.requires_grad_(True)
            print(f"  randn(3) + requires_grad_(True) => shape={x.shape}, grad={x.requires_grad}")
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")
test("API Check: Tensor.randn signature", t12)

print("\n=== ALL TESTS COMPLETE ===")

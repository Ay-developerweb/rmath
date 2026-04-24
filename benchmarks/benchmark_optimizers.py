import rmath.array as ra
import torch
import time

def benchmark_optimizers():
    # Model size: 1000x1000 weight matrix + 1000 bias
    rows, cols = 1000, 1000
    print(f"── OPTIMIZER BENCHMARK ({rows}x{cols} parameters) ──")
    
    # 1. ADAM BENCHMARK
    print("\n[ADAM]")
    params_rm = [ra.Tensor.randn(rows, cols, requires_grad=True), ra.Tensor.randn(cols, requires_grad=True)]
    # Mock gradients
    params_rm[0].grad = ra.Array.randn(rows, cols)
    params_rm[1].grad = ra.Array.randn(cols)
    opt_rm = ra.Adam(params_rm, lr=0.001)

    params_pt = [torch.randn(rows, cols, dtype=torch.float64, requires_grad=True), 
                 torch.randn(cols, dtype=torch.float64, requires_grad=True)]
    params_pt[0].grad = torch.randn(rows, cols, dtype=torch.float64)
    params_pt[1].grad = torch.randn(cols, dtype=torch.float64)
    opt_pt = torch.optim.Adam(params_pt, lr=0.001)

    # Benchmark RM
    start = time.perf_counter()
    for _ in range(100):
        opt_rm.step()
    end = time.perf_counter()
    rm_time = (end - start) / 100 * 1000
    print(f"  Rmath Adam.step():  {rm_time:7.3f} ms")

    # Benchmark PT
    start = time.perf_counter()
    for _ in range(100):
        opt_pt.step()
    end = time.perf_counter()
    pt_time = (end - start) / 100 * 1000
    print(f"  PyTorch Adam.step(): {pt_time:7.3f} ms")
    print(f"  Speedup vs PT:      {pt_time/rm_time:5.1f}x")

    # 2. SGD BENCHMARK
    print("\n[SGD with Momentum]")
    params_rm = [ra.Tensor.randn(rows, cols, requires_grad=True)]
    params_rm[0].grad = ra.Array.randn(rows, cols)
    opt_rm = ra.SGD(params_rm, lr=0.01, momentum=0.9)

    params_pt = [torch.randn(rows, cols, dtype=torch.float64, requires_grad=True)]
    params_pt[0].grad = torch.randn(rows, cols, dtype=torch.float64)
    opt_pt = torch.optim.SGD(params_pt, lr=0.01, momentum=0.9)

    # Benchmark RM
    start = time.perf_counter()
    for _ in range(100):
        opt_rm.step()
    end = time.perf_counter()
    rm_time = (end - start) / 100 * 1000
    print(f"  Rmath SGD.step():   {rm_time:7.3f} ms")

    # Benchmark PT
    start = time.perf_counter()
    for _ in range(100):
        opt_pt.step()
    end = time.perf_counter()
    pt_time = (end - start) / 100 * 1000
    print(f"  PyTorch SGD.step():  {pt_time:7.3f} ms")
    print(f"  Speedup vs PT:      {pt_time/rm_time:5.1f}x")

if __name__ == "__main__":
    benchmark_optimizers()

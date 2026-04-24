import rmath
from rmath.vector import Vector
from rmath.array import Array

# 1. Test Small Matrix (Inline)
m = Array([[1, 2], [3, 4]])
v = Vector([10, 20])
res = m + v
print(f"Matrix + Vector Broadcasting (Inline):\n{res.to_list()}")
# Expected: [[11, 22], [13, 24]]

# 2. Test Large Matrix (Heap + faer)
m_large = Array.ones(100, 100)
v_large = Vector([1.0] * 100)
res_large = m_large + v_large
print(f"Large Matrix (100x100) + Vector Sum: {sum(sum(r) for r in res_large.to_list())}")
# Expected: 100*100*(1+1) = 20000

# 3. Test Matmul (faer)
m1 = Array([[1, 2], [3, 4]])
m2 = Array([[5, 6], [7, 8]])
res_mul = m1 @ m2
print(f"Matrix Multiplication:\n{res_mul.to_list()}")
# Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

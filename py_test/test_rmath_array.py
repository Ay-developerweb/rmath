import pytest
import numpy as np
from rmath.array import Array
from rmath.vector import Vector
import rmath.array as ra

def test_array_creation():
    data = [[1.0, 2.0], [3.0, 4.0]]
    arr = Array(data)
    assert arr.to_list() == data
    assert arr.shape() == (2, 2)

def test_array_factory_methods():
    z = Array.zeros(3, 3)
    assert z.to_list() == [[0.0, 0.0, 0.0]] * 3
    
    o = Array.ones(2, 4)
    assert o.to_list() == [[1.0] * 4] * 2
    
    e = Array.eye(3)
    assert e.to_list() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

def test_array_matmul():
    m1 = Array([[1.0, 2.0], [3.0, 4.0]])
    m2 = Array([[5.0, 6.0], [7.0, 8.0]])
    res = m1 @ m2
    # Expected: [[19.0, 22.0], [43.0, 50.0]]
    assert res.to_list() == [[19.0, 22.0], [43.0, 50.0]]

def test_array_broadcasting():
    m = Array([[1.0, 2.0], [3.0, 4.0]])
    v = Vector([10.0, 20.0])
    res = m + v
    assert res.to_list() == [[11.0, 22.0], [13.0, 24.0]]
    
    # Scalar broadcasting
    res_s = m + 100.0
    assert res_s.to_list() == [[101.0, 102.0], [103.0, 104.0]]
    
    # Reflected scalar
    res_rs = 100.0 + m
    assert res_rs.to_list() == [[101.0, 102.0], [103.0, 104.0]]

def test_array_linear_algebra():
    m = Array([[1.0, 2.0], [3.0, 4.0]])
    det = m.det()
    # det = 1*4 - 2*3 = -2
    assert det == pytest.approx(-2.0)
    
    inv = m.inv()
    ident = m @ inv
    # check if identity matrix (approx)
    for i, row in enumerate(ident.to_list()):
        for j, val in enumerate(row):
            if i == j: assert val == pytest.approx(1.0)
            else: assert val == pytest.approx(0.0)

def test_array_tools():
    m = Array([[1.0, 2.0], [3.0, 4.0]])
    assert m.transpose().to_list() == [[1.0, 3.0], [2.0, 4.0]]
    
    # Sum axis=0 (across rows -> cols)
    assert m.sum(axis=0) == pytest.approx([4.0, 6.0])
    # Sum axis=1 (across columns -> rows)
    assert m.sum(axis=1) == pytest.approx([3.0, 7.0])
    # Sum axis=None (total)
    assert m.sum() == pytest.approx(10.0)

def test_array_from_numpy():
    data = np.random.rand(10, 10)
    ra_arr = Array.from_numpy(data)
    assert ra_arr.shape() == (10, 10)
    assert ra_arr.to_list()[0][0] == pytest.approx(data[0, 0])

def test_array_sigmoid():
    m = Array([[0.0, 1.0], [-1.0, 100.0]])
    res = m.sigmoid()
    assert res.to_list()[0][0] == pytest.approx(0.5)
    assert res.to_list()[0][1] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))
    assert res.to_list()[1][1] == pytest.approx(1.0) # approx

import pytest
import math
from rmath.vector import Vector
import rmath.vector as rv

def test_vector_creation():
    v = Vector([1.0, 2.0, 3.0])
    assert v.tolist() == [1.0, 2.0, 3.0]
    assert len(v) == 3

def test_vector_arithmetic_scalar():
    v = Vector([1.0, 2.0, 3.0])
    assert (v + 10).tolist() == [11.0, 12.0, 13.0]
    assert (v - 1).tolist() == [0.0, 1.0, 2.0]
    assert (v * 2).tolist() == [2.0, 4.0, 6.0]
    assert (v / 2).tolist() == [0.5, 1.0, 1.5]

def test_vector_arithmetic_reflected():
    v = Vector([1.0, 2.0, 4.0])
    assert (10 + v).tolist() == [11.0, 12.0, 14.0]
    assert (10 - v).tolist() == [9.0, 8.0, 6.0]
    assert (2 * v).tolist() == [2.0, 4.0, 8.0]
    assert (4 / v).tolist() == [4.0, 2.0, 1.0]

def test_vector_arithmetic_vector():
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([10.0, 20.0, 30.0])
    assert (v1 + v2).tolist() == [11.0, 22.0, 33.0]
    assert (v2 - v1).tolist() == [9.0, 18.0, 27.0]

def test_vector_negation():
    v = Vector([1.0, -2.0, 0.0])
    assert (-v).tolist() == [-1.0, 2.0, -0.0]

def test_vector_range():
    v = rv.range(0, 5, 1)
    assert v.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    
    # Test large range (Heap)
    v_large = rv.range(0, 100, 1)
    assert len(v_large) == 100
    assert v_large.tolist()[99] == 99.0

def test_vector_linspace():
    v = rv.linspace(0, 1, 5)
    assert v.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]

def test_vector_map():
    v = Vector([1.0, 4.0, 9.0])
    # Note: map takes a python callable, which is slower but flexible
    res = v.map(math.sqrt)
    assert res.tolist() == [1.0, 2.0, 3.0]

def test_vector_zip_map():
    v1 = Vector([1.0, 2.0, 3.0])
    v2 = Vector([4.0, 5.0, 6.0])
    res = v1.zip_map(v2, lambda x, y: x * y)
    assert res.tolist() == [4.0, 10.0, 18.0]

def test_vector_error_handling():
    v1 = Vector([1.0, 2.0])
    v2 = Vector([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same length"):
        v1 + v2

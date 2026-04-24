import rmath
import math
import unittest

class TestVector(unittest.TestCase):
    def setUp(self):
        self.data = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.v = rmath.vector.Vector(self.data)

    def test_initialization(self):
        self.assertEqual(self.v.len(), 5)
        self.assertEqual(self.v.to_list(), self.data)

    def test_arithmetic(self):
        # Chaining: (x + 10) * 2 / 2 - 10 == x
        res = self.v.add_scalar(10.0).mul_scalar(2.0).div_scalar(2.0).sub_scalar(10.0).to_list()
        for i in range(len(self.data)):
            self.assertAlmostEqual(res[i], self.data[i])

    def test_trigonometry(self):
        # v.sin()
        res = self.v.sin().to_list()
        for i in range(len(self.data)):
            self.assertAlmostEqual(res[i], math.sin(self.data[i]))

    def test_reductions(self):
        # sum
        self.assertAlmostEqual(self.v.sum(), sum(self.data))
        # mean
        self.assertAlmostEqual(self.v.mean(), sum(self.data) / len(self.data))
        # variance
        # Standard Var: 2.5
        self.assertAlmostEqual(self.v.variance(), 2.5)

    def test_rounding(self):
        v_dec = rmath.vector.Vector([1.1, 2.5, 3.9])
        self.assertEqual(v_dec.ceil().to_list(), [2.0, 3.0, 4.0])
        self.assertEqual(v_dec.floor().to_list(), [1.0, 2.0, 3.0])

    def test_legacy_api(self):
        # rmath.vector.sin([list])
        res = rmath.vector.sin(self.data)
        for i in range(len(self.data)):
            self.assertAlmostEqual(res[i], math.sin(self.data[i]))

if __name__ == '__main__':
    unittest.main()

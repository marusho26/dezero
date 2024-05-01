import unittest
import step09 as s
import numpy as np

def numerical_diff(f, x, eps=1e-4):
    x0 = s.Variable(x.data - eps)
    x1 = s.Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = s.Variable(np.array(3.0))
        y = s.square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    def test_grandient_check(self):
        x = s.Variable(np.random.rand(1))
        y = s.square(x)
        y.backward()
        num_grad = numerical_diff(s.square, x)
        flg = np.allclose(x.grad, num_grad)
        print(x.grad, ':', num_grad)
        self.assertTrue(flg)


unittest.main()
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from nnlib.layers import ReLU, LeakyReLU, PReLU
from nnlib.utils import numerical_grad

# don't use zeros as inputs since this introduces the kink in
# the function (i.e. the function is non-differentiable at x=0)
X = np.array([
    [1, -2, 7],
    [-1, 1, -5]], dtype=float)

grad_top = np.ones(X.shape)


class ReLUTest(TestCase):

    def setUp(self):
        self.X = X.copy()

        self.expected_Z = np.array([
            [1, 0, 7],
            [0, 1, 0]], dtype=float)

        self.grad_top = grad_top.copy()
        self.layer = ReLU()

    def test_forward(self):
        Z = self.layer.forward(self.X)
        assert_array_equal(Z, self.expected_Z)
        assert_array_equal(self.layer._X_cache, self.X)

    def test_backward(self):
        self.layer.forward(self.X)
        d_X = self.layer.backward(self.grad_top)

        assert_array_almost_equal(
            numerical_grad(self.layer.forward, self.X),
            d_X
        )


class LeakyReLUTest(TestCase):

    def setUp(self):
        self.X = X.copy()

        self.expected_Z = np.array([
            [1, -0.02, 7],
            [-0.01, 1, -0.05]], dtype=float)

        self.grad_top = grad_top.copy()
        self.layer = LeakyReLU()

    def test_forward(self):
        Z = self.layer.forward(self.X)
        assert_array_almost_equal(Z, self.expected_Z)

    def test_backward(self):
        self.layer.forward(self.X)
        d_X = self.layer.backward(self.grad_top)

        assert_array_almost_equal(
            numerical_grad(self.layer.forward, self.X),
            d_X
        )


class PReLUTest(TestCase):

    def setUp(self):
        self.X = X.copy()

        self.expected_Z = np.array([
            [1, -0.1, 7],
            [-0.05, 1, -0.25]], dtype=float)

        self.grad_top = grad_top.copy()
        self.layer = PReLU()
        self.leakiness = 0.05
        self.layer.leakiness = self.leakiness

    def test_forward(self):
        Z = self.layer.forward(self.X)
        assert_array_almost_equal(Z, self.expected_Z)

    def test_grad_on_leakiness(self):
        self.layer.forward(self.X)
        self.layer.backward(self.grad_top)
        d_leakiness = self.layer.d_leakiness

        layer = self.layer

        def forward_as_func_of_leakiness(leakiness_):
            layer.leakiness = leakiness_
            return layer.forward(self.X)

        self.assertAlmostEqual(
            numerical_grad(forward_as_func_of_leakiness, self.leakiness),
            d_leakiness
        )

    def test_grad_on_X(self):
        self.layer.forward(self.X)
        d_X = self.layer.backward(self.grad_top)

        assert_array_almost_equal(
            numerical_grad(self.layer.forward, self.X),
            d_X
        )

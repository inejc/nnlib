from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from nnlib.layers import FullyConnected
from nnlib.utils import numerical_grad


class FullyConnectedTest(TestCase):
    input_dim_1d = 3
    num_neurons = 4

    def setUp(self):
        self.X = np.array([
            [1, -2, 1],
            [0, 1, 5]], dtype=float)

        self.expected_Z = np.array([
            [6, 5, 7, 4],
            [-1, -18, 17, 27]], dtype=float)

        self.grad_top = np.ones(self.expected_Z.shape)

        self.W = np.array([
            [1, -1, 5, 0],
            [-2, -4, 1, 1],
            [0, -3, 3, 5]], dtype=float)

        self.b = np.ones((1, self.num_neurons))
        self.layer = FullyConnected(self.input_dim_1d, self.num_neurons)
        self.layer.W = self.W
        self.layer.b = self.b

    def test_forward(self):
        Z = self.layer.forward(self.X)
        assert_array_equal(Z, self.expected_Z)
        assert_array_equal(self.layer._X_cache, self.X)

    # test backward
    def test_grad_on_W(self):
        self.layer.forward(self.X)
        self.layer.backward(self.grad_top)
        d_W = self.layer.d_W

        layer = self.layer

        def forward_as_func_of_W(W_):
            layer.W = W_
            return layer.forward(self.X)

        assert_array_almost_equal(
            numerical_grad(forward_as_func_of_W, self.W),
            d_W
        )

    def test_grad_on_b(self):
        self.layer.forward(self.X)
        self.layer.backward(self.grad_top)
        d_b = self.layer.d_b

        layer = self.layer

        def forward_as_func_of_b(b_):
            layer.b = b_
            return layer.forward(self.X)

        assert_array_almost_equal(
            numerical_grad(forward_as_func_of_b, self.b),
            d_b
        )

    def test_grad_on_X(self):
        self.layer.forward(self.X)
        d_X = self.layer.backward(self.grad_top)

        assert_array_almost_equal(
            numerical_grad(self.layer.forward, self.X),
            d_X
        )

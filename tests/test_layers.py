from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from nnlib.layers import FullyConnected
from nnlib.utils import numerical_grad


class FullyConnectedLayerTest(TestCase):
    input_dim_1d = 3
    num_neurons = 4

    W = np.array([
        [1, -1, 5, 0],
        [-2, -4, 1, 1],
        [0, -3, 3, 5]], dtype=float)

    b = np.ones((1, num_neurons))

    input_ = np.array([
        [1, -2, 1],
        [0, 1, 5]], dtype=float)

    expected_output = np.array([
        [6, 5, 7, 4],
        [-1, -18, 17, 27]], dtype=float)

    grad_top = np.ones(expected_output.shape)

    def setUp(self):
        self.layer = FullyConnected(self.input_dim_1d, self.num_neurons)
        self.layer.W = self.W
        self.layer.b = self.b

    def test_forward(self):
        z = self.layer.forward(self.input_)
        assert_array_equal(z, self.expected_output)
        assert_array_equal(self.layer.input_cache, self.input_)

    # test backward
    def test_grad_on_w(self):
        self.layer.forward(self.input_)
        self.layer.backward(self.grad_top)
        grad_w = self.layer.d_W

        layer = self.layer

        def forward_as_func_of_w(W_):
            layer.W = W_
            return layer.forward(self.input_)

        assert_array_almost_equal(
            numerical_grad(forward_as_func_of_w, self.W),
            grad_w
        )

    def test_grad_on_b(self):
        self.layer.forward(self.input_)
        self.layer.backward(self.grad_top)
        grad_b = self.layer.d_b

        layer = self.layer

        def forward_as_func_of_b(b_):
            layer.b = b_
            return layer.forward(self.input_)

        assert_array_almost_equal(
            numerical_grad(forward_as_func_of_b, self.b),
            grad_b
        )

    def test_grad_on_input(self):
        self.layer.forward(self.input_)
        grad_input = self.layer.backward(self.grad_top)

        assert_array_almost_equal(
            numerical_grad(self.layer.forward, self.input_),
            grad_input
        )

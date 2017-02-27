from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nnlib.layers import SoftmaxWithCrossEntropy
from nnlib.utils import numerical_grad


class SoftmaxWithCrossEntropyLayerTest(TestCase):
    X = np.array([
        [1, -2, 1],
        [0, 1, 5]], dtype=float)

    y = np.array([0, 1])

    expected_probs = np.array([
        [0.48785555, 0.0242889, 0.48785555],
        [0.00657326, 0.0178679, 0.97555875]], dtype=float)

    expected_loss = 2.37124

    def setUp(self):
        self.layer = SoftmaxWithCrossEntropy()

    def test_forward(self):
        loss = self.layer.forward(self.X, self.y)

        self.assertAlmostEqual(loss, self.expected_loss, places=6)
        assert_array_almost_equal(self.layer.probs_cache, self.expected_probs)
        assert_array_equal(self.layer.y_cache, self.y)

    def test_backward(self):
        self.layer.forward(self.X, self.y)
        d_X = self.layer.backward()

        layer = self.layer
        y = self.y

        def forward_as_func_of_X(X):
            return layer.forward(X, y)

        assert_array_almost_equal(
            numerical_grad(forward_as_func_of_X, self.X),
            d_X
        )

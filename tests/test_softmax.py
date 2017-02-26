from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from nnlib.layers import Softmax
from nnlib.utils import numerical_grad


class SoftmaxLayerTest(TestCase):
    input_ = np.array([
        [1, -2, 1],
        [0, 1, 5]], dtype=float)

    expected_output = np.array([
        [0.48785555, 0.0242889, 0.48785555],
        [0.00657326, 0.0178679, 0.97555875]], dtype=float)

    classes_sums = np.ones((2,), dtype=float)

    grad_top = np.ones(expected_output.shape)

    def setUp(self):
        self.layer = Softmax()

    def test_forward(self):
        output = self.layer.forward(self.input_)
        assert_array_almost_equal(output, self.expected_output)
        assert_array_almost_equal(np.sum(output, axis=1), self.classes_sums)
        assert_array_almost_equal(self.layer.output_cache, self.expected_output)

    def test_backward(self):
        pass
        # self.layer.forward(self.input_)
        # d_input = self.layer.backward(self.grad_top)
        #
        # assert_array_almost_equal(
        #     numerical_grad(self.layer.forward, self.input_),
        #     d_input
        # )

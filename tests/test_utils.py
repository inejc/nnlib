from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from nnlib.utils import numerical_grad


class UtilsTest(TestCase):

    @staticmethod
    def test_numerical_grad_ndarray():
        X = np.array([
            [-5, 1, -1, 10, -2],
            [8, 10, -12, 3, 1],
            [0, 0, 2, -1, 5]], dtype=float)

        Y = np.array([
            [1, 2, 2, 1],
            [-1, -2, -1, -1],
            [4, 5, 1, -2],
            [8, -10, 12, 1],
            [0, 10, -1, 2]], dtype=float)

        def dot_Y(X_):
            return np.dot(X_, Y)

        expected_grad_X = np.array([
            [6, -5, 8, 11, 11],
            [6, -5, 8, 11, 11],
            [6, -5, 8, 11, 11]], dtype=float)

        assert_array_almost_equal(
            numerical_grad(dot_Y, X),
            expected_grad_X
        )

    def test_numerical_grad_scalar(self):

        def times_5(x_):
            return x_ * 5

        self.assertAlmostEqual(numerical_grad(times_5, 12), 5)

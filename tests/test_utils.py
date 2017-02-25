from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from nnlib.utils import numerical_grad


class UtilsTest(TestCase):

    @staticmethod
    def test_numerical_grad():
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

        def dot_y(X_):
            return np.dot(X_, Y)

        expected_grad_x = np.array([
            [6, -5, 8, 11, 11],
            [6, -5, 8, 11, 11],
            [6, -5, 8, 11, 11]], dtype=float)

        assert_array_almost_equal(
            numerical_grad(dot_y, X),
            expected_grad_x
        )

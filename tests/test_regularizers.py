from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from nnlib.regularizers import l2
from tests.utils import numerical_grad


class L2Test(TestCase):

    def setUp(self):
        self.W = np.array([
            [1, -1, 5, 0],
            [-2, -4, 1, 1],
            [0, -3, 3, 5]], dtype=float)

    def test_L2(self):
        regularizer = l2(lambda_=0)
        self.assertEqual(regularizer.loss(self.W), 0)
        regularizer = l2(lambda_=2)
        self.assertEqual(regularizer.loss(self.W), 92)

    def test_L2_grad(self):
        regularizer = l2(lambda_=0)
        assert_array_almost_equal(
            numerical_grad(regularizer.loss, self.W),
            regularizer.grad(self.W)
        )

        regularizer = l2(lambda_=2)
        assert_array_almost_equal(
            numerical_grad(regularizer.loss, self.W),
            regularizer.grad(self.W)
        )

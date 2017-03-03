from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from nnlib.utils import classification_accuracy
from nnlib.utils import numerical_grad, yield_data_in_batches


class UtilsTest(TestCase):

    def setUp(self):
        self.Y = np.array([
            [1, 2, 2, 1],
            [-1, -2, -1, -1],
            [4, 5, 1, -2],
            [8, -10, 12, 1],
            [0, 10, -1, 2]], dtype=float)

    def test_numerical_grad_ndarray(self):
        X = np.array([
            [-5, 1, -1, 10, -2],
            [8, 10, -12, 3, 1],
            [0, 0, 2, -1, 5]], dtype=float)

        def dot_Y(X_):
            return np.dot(X_, self.Y)

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

    def test_yield_data_in_batches_no_shuffle(self):
        batches = yield_data_in_batches(batch_size=2, X=self.Y, shuffle=False)

        for i, batch in enumerate(batches):
            if i == 0:
                expected_batch = np.array([
                    [1, 2, 2, 1],
                    [-1, -2, -1, -1]], dtype=float)
                assert_array_equal(batch, expected_batch)
            elif i == 1:
                expected_batch = np.array([
                    [4, 5, 1, -2],
                    [8, -10, 12, 1]], dtype=float)
                assert_array_equal(batch, expected_batch)
            else:
                expected_batch = np.array([
                    [0, 10, -1, 2]], dtype=float)
                assert_array_equal(batch, expected_batch)

    def test_yield_data_in_batches_shuffle(self):
        batches = yield_data_in_batches(batch_size=2, X=self.Y)

        for i, batch in enumerate(batches):
            if i == 0 or i == 1:
                self.assertEqual(batch.shape, (2, 4))
            else:
                self.assertEqual(batch.shape, (1, 4))

    def test_yield_data_in_batches_shuffle_with_y(self):
        y = np.arange(self.Y.shape[0])
        batches = yield_data_in_batches(batch_size=2, X=self.Y, y=y)

        for i, batch in enumerate(batches):
            X_batch, y_batch = batch
            assert_array_equal(X_batch, self.Y[y_batch])

    def test_classification_accuracy(self):
        y = np.array([3, 0, 0, 1])
        y_pred = np.array([3, 1, 0, 2])
        acc = classification_accuracy(y, y_pred)
        self.assertEqual(acc, 0.5)

from unittest import TestCase

import numpy as np

from nnlib import Model
from nnlib.layers import FullyConnected, ReLU, PReLU, SoftmaxWithCrossEntropy
from nnlib.optimizers import SGD


class ModelTest(TestCase):

    def setUp(self):
        self.n = 50  # number of examples (data points)
        self.d = 2  # number of features (dimensionality of the data)
        self.k = 2  # number of classes
        self.h = 50  # number of neurons in the hidden layer
        self.X = np.random.randn(self.n, self.d)
        self.model = Model()

    def test_add_layers(self):
        self.assertEqual(len(self.model._layers), 0)
        self.model.add(ReLU())
        self.assertEqual(len(self.model._layers), 1)
        self.model.add(ReLU())
        self.assertEqual(len(self.model._layers), 2)
        self.model.add(ReLU())

    def test_compile(self):
        optimizer = SGD()
        self.assertIsNone(self.model._optimizer)

        self.model.add(ReLU())
        self.model.add(PReLU())
        self.model.compile(optimizer)

        self.assertEqual(self.model._optimizer, optimizer)
        self.assertEqual(len(self.model._optimizer._layers), 1)

    def test_predict(self):
        self.model.add(FullyConnected(self.d, self.h))
        self.model.add(PReLU())
        self.model.add(FullyConnected(self.h, self.k))
        self.model.add(SoftmaxWithCrossEntropy())

        y_pred = self.model.predict(self.X)
        self.assertEqual(y_pred.shape, (self.n,))

    def test_predict_proba(self):
        self.model.add(FullyConnected(self.d, self.h))
        self.model.add(PReLU())
        self.model.add(FullyConnected(self.h, self.k))
        self.model.add(SoftmaxWithCrossEntropy())

        probs = self.model.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n, self.k))

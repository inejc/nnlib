from unittest import TestCase

import numpy as np

from nnlib import Model
from nnlib.layers import FullyConnected, ReLU, SoftmaxWithCrossEntropy
from nnlib.optimizers import SGD
from nnlib.utils import classification_accuracy


class XorDataTest(TestCase):

    def test_training_accuracy(self):
        n = 50   # number of examples (data points)
        d = 2    # number of features (dimensionality of the data)
        k = 2    # number of classes
        h = 50   # number of neurons in the hidden layer

        np.random.seed(0)
        X = np.random.randn(n, d)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

        model = Model()
        model.add(FullyConnected(num_input_neurons=d, num_neurons=h))
        model.add(ReLU())
        model.add(FullyConnected(num_input_neurons=h, num_neurons=k))
        model.add(SoftmaxWithCrossEntropy())
        model.compile(SGD(lr=1))

        model.train(X, y, batch_size=n, num_epochs=180)
        y_pred = model.predict(X)

        acc = classification_accuracy(y, y_pred)
        self.assertEqual(acc, 1)

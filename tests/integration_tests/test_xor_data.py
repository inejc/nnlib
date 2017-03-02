from unittest import TestCase

import numpy as np

from nnlib import Model
from nnlib.layers import FullyConnected, ReLU, SoftmaxWithCrossEntropy
from nnlib.optimizers import SGD


class XorDataTest(TestCase):

    def test_training_accuracy(self):
        N = 50   # number of examples (data points)
        D = 2    # number of features (dimensionality of the data)
        K = 2    # number of classes
        H = 50   # number of neurons in the hidden layer

        np.random.seed(0)
        X = np.random.randn(N, D)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

        model = Model()
        model.add(FullyConnected(input_dim_1d=D, num_neurons=H))
        model.add(ReLU())
        model.add(FullyConnected(input_dim_1d=H, num_neurons=K))
        model.add(SoftmaxWithCrossEntropy())
        model.compile(SGD(lr=1))

        model.train(X, y, batch_size=32, num_epochs=200)
        probs = model.predict(X)

        y_pred = np.argmax(probs, axis=1)
        acc = np.mean(np.equal(y_pred, y))
        self.assertEqual(acc, 1)

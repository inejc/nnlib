from unittest import TestCase

import numpy as np

from nnlib import Model
from nnlib.layers import FullyConnected, ReLU, SoftmaxWithCrossEntropy
from nnlib.optimizers import SGD
from nnlib.utils import classification_accuracy
from tests.utils import xor_data


class XorDataTest(TestCase):

    def test_training_accuracy(self):
        n = 50   # number of examples (data points)
        d = 2    # number of features (dimensionality of the data)
        h = 50   # number of neurons in the hidden layer
        k = 2    # number of classes

        np.random.seed(0)
        X, y = xor_data(num_examples=n)

        model = Model()
        model.add(FullyConnected(num_input_neurons=d, num_neurons=h))
        model.add(ReLU())
        model.add(FullyConnected(num_input_neurons=h, num_neurons=k))
        model.add(SoftmaxWithCrossEntropy())
        model.compile(SGD(lr=1))

        model.fit(X, y, batch_size=n, num_epochs=180)
        y_pred = model.predict(X)

        acc = classification_accuracy(y, y_pred)
        self.assertEqual(acc, 1)

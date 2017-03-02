from unittest import TestCase

from nnlib.layers import Layer
from nnlib.optimizers import SGD, ParamsGrads


class DummyLayer(Layer):

    def __init__(self):
        self.dummy_param0 = 10
        self.dummy_grad0 = 2

        self.dummy_param1 = 5
        self.dummy_grad1 = 1

    def forward(self):
        pass

    def backward(self):
        pass

    def updatable_params_grads_names(self):
        return [
            ParamsGrads(params='dummy_param0', grads='dummy_grad0'),
            ParamsGrads(params='dummy_param1', grads='dummy_grad1')
        ]


class SGDTest(TestCase):

    def setUp(self):
        self.sgd = SGD(lr=0.5)

    def test_register_layer(self):
        self.assertEqual(len(self.sgd.layers), 0)
        self.sgd.register_layer(DummyLayer())
        self.assertEqual(len(self.sgd.layers), 1)
        self.sgd.register_layer(DummyLayer())
        self.assertEqual(len(self.sgd.layers), 2)

    def test_make_updates(self):
        layer = DummyLayer()
        self.sgd.register_layer(layer)
        self.sgd.make_updates()
        self.assertEqual(layer.dummy_param0, 9)
        self.assertEqual(layer.dummy_param1, 4.5)

from unittest import TestCase

from nnlib.layers import Layer
from nnlib.optimizers import SGD, ParamGradNames


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
            ParamGradNames(param_name='dummy_param0', grad_name='dummy_grad0'),
            ParamGradNames(param_name='dummy_param1', grad_name='dummy_grad1')
        ]


class SGDTest(TestCase):

    def setUp(self):
        self.sgd = SGD(lr=0.5)
        self.layer = DummyLayer()

    def test_register_layer(self):
        self.assertEqual(len(self.sgd._layers), 0)
        self.sgd.register_layer(self.layer)
        self.assertEqual(len(self.sgd._layers), 1)
        self.sgd.register_layer(self.layer)
        self.assertEqual(len(self.sgd._layers), 2)

    def test_make_updates(self):
        self.sgd.register_layer(self.layer)
        self.sgd.make_updates()
        self.assertEqual(self.layer.dummy_param0, 9)
        self.assertEqual(self.layer.dummy_param1, 4.5)

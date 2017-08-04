from unittest import TestCase

from nnlib.layers import Layer, ParamGradNames
from nnlib.optimizers import SGD, SGDMomentum


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

    def get_updatable_params_grads_names(self):
        return [
            ParamGradNames(param_name='dummy_param0', grad_name='dummy_grad0'),
            ParamGradNames(param_name='dummy_param1', grad_name='dummy_grad1')
        ]


class SGDTest(TestCase):

    def setUp(self):
        self.sgd = SGD(lr=0.5)
        self.sgd_m = SGDMomentum(lr=0.5, nesterov=False)
        self.sgd_m_n = SGDMomentum(lr=0.5)
        self.layer = DummyLayer()

    def test_register_layer_sgd(self):
        self.assertEqual(len(self.sgd._layers), 0)
        self.sgd.register_layer(self.layer)
        self.assertEqual(len(self.sgd._layers), 1)
        self.sgd.register_layer(self.layer)
        self.assertEqual(len(self.sgd._layers), 2)

    def test_register_layer_sgd_m(self):
        self.assertEqual(len(self.sgd_m._layers_caches), 0)
        self.sgd_m.register_layer(self.layer)
        self.assertEqual(len(self.sgd_m._layers_caches), 1)
        self.sgd_m.register_layer(self.layer)
        self.assertEqual(len(self.sgd_m._layers_caches), 2)

    def test_register_layer_sgd_m_n(self):
        self.assertEqual(len(self.sgd_m_n._layers_caches), 0)
        self.sgd_m_n.register_layer(self.layer)
        self.assertEqual(len(self.sgd_m_n._layers_caches), 1)
        self.sgd_m_n.register_layer(self.layer)
        self.assertEqual(len(self.sgd_m_n._layers_caches), 2)

    def test_make_updates_sgd(self):
        self.sgd.register_layer(self.layer)
        self.sgd.update_layers()
        self.assertEqual(self.layer.dummy_param0, 9)
        self.assertEqual(self.layer.dummy_param1, 4.5)

    def test_make_updates_sgd_m(self):
        self.sgd_m.register_layer(self.layer)
        self.sgd_m.update_layers()
        self.assertEqual(self.layer.dummy_param0, 9)
        self.assertEqual(self.layer.dummy_param1, 4.5)
        self.sgd_m.update_layers()
        self.assertEqual(self.layer.dummy_param0, 7.1)
        self.assertEqual(self.layer.dummy_param1, 3.55)

    def test_make_updates_sgd_m_n(self):
        self.sgd_m_n.register_layer(self.layer)
        self.sgd_m_n.update_layers()
        self.assertEqual(self.layer.dummy_param0, 8.1)
        self.assertEqual(self.layer.dummy_param1, 4.05)
        self.sgd_m_n.update_layers()
        self.assertEqual(self.layer.dummy_param0, 5.39)
        self.assertEqual(self.layer.dummy_param1, 2.695)

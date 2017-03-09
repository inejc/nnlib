from unittest import TestCase

from nnlib.layers import Layer


class SimpleDummyLayer(Layer):

    def forward(self):
        pass

    def backward(self):
        pass


class FullDummyLayer(Layer):

    def forward(self):
        pass

    def backward(self):
        pass

    def get_regularization_loss(self):
        pass

    def get_updatable_params_grads_names(self):
        pass


class LayerTest(TestCase):

    def test_is_regularized(self):
        layer = SimpleDummyLayer()
        self.assertFalse(layer.is_regularized())

        layer = FullDummyLayer()
        self.assertTrue(layer.is_regularized())

    def test_has_updatable_params(self):
        layer = SimpleDummyLayer()
        self.assertFalse(layer.has_updatable_params())

        layer = FullDummyLayer()
        self.assertTrue(layer.has_updatable_params())

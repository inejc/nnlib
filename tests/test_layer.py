from unittest import TestCase

from nnlib.layers import Layer


class DummyLayerNoUpdatableParams(Layer):

    def forward(self):
        pass

    def backward(self):
        pass


class DummyLayerUpdatableParams(Layer):

    def forward(self):
        pass

    def backward(self):
        pass

    def updatable_params_grads_names(self):
        pass


class LayerTest(TestCase):

    def test_has_updatable_params(self):
        layer = DummyLayerNoUpdatableParams()
        self.assertFalse(layer.has_updatable_params)

        layer = DummyLayerUpdatableParams()
        self.assertTrue(layer.has_updatable_params)

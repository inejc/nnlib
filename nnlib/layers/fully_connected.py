import numpy as np

from nnlib.layers import Layer, ParamGradNames


class FullyConnected(Layer):
    """A fully connected or dense layer.

    Parameters
    ----------
    num_input_neurons: int
        Number of input neurons (i.e. the dimensionality of the input
        data).

    num_neurons: int
        Number of neurons in this layer (i.e. output dimensionality).

    regularizer: nnlib.regularizers.Regularizer, default None
        Weight regularizer (i.e. regularizer that computes the weights
        penalty).
    """

    def __init__(self, num_input_neurons, num_neurons, regularizer=None):
        self.W = 0.01 * np.random.rand(num_input_neurons, num_neurons)
        self.b = np.zeros((1, num_neurons))

        self._X_cache = None
        self.d_W = None
        self.d_b = None

        self._regularizer = regularizer

    def forward(self, X):
        # cache the input so that we can use it at the
        # backward pass when computing the gradient on W
        self._X_cache = X

        Z = np.dot(X, self.W) + self.b
        return Z

    def backward(self, grad_top):
        self.d_W = np.dot(self._X_cache.T, grad_top)

        # add the regularization gradient if regularization is enabled
        if self._regularizer is not None:
            self.d_W += self._regularizer.grad(self.W)

        self.d_b = np.sum(grad_top, axis=0, keepdims=True)

        # the gradient on input is the new gradient from the
        # top for the next layer during the backward pass
        d_X = np.dot(grad_top, self.W.T)
        return d_X

    def get_regularization_loss(self):
        if self._regularizer is None:
            return 0

        return self._regularizer.loss(self.W)

    def get_updatable_params_grads_names(self):
        return [
            ParamGradNames(param_name='W', grad_name='d_W'),
            ParamGradNames(param_name='b', grad_name='d_b')
        ]

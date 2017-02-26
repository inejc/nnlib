import numpy as np


class FullyConnected(object):
    """A fully connected or dense layer.

    Parameters
    ----------
    input_dim_1d: int
        Number of input neurons i.e. input dimensionality.

    num_neurons: int
        Number of neurons in this layer i.e. output dimensionality.
    """

    def __init__(self, input_dim_1d, num_neurons):
        self.W = 0.01 * np.random.rand(input_dim_1d, num_neurons)
        self.b = np.zeros((1, num_neurons))

        self.input_cache = None
        self.d_W = None
        self.d_b = None

    def forward(self, input_):
        # cache the input so that we can use it at the
        # backward pass when computing the gradient on W
        self.input_cache = input_

        output = np.dot(input_, self.W) + self.b
        return output

    def backward(self, grad_top):
        self.d_W = np.dot(self.input_cache.T, grad_top)
        self.d_b = np.sum(grad_top, axis=0, keepdims=True)

        # the gradient on input is the new gradient from the
        # top for the next layer during the backward pass
        d_input = np.dot(grad_top, self.W.T)
        return d_input

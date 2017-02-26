import numpy as np


class Softmax(object):
    """A softmax or normalized exponential function layer."""

    def __init__(self):
        self.output_cache = None

    def forward(self, input_):
        # shift the input so that the highest value is
        # zero (numerical stability)
        input_ -= np.max(input_, axis=1).reshape((-1, 1))

        output = np.exp(input_)
        output /= np.sum(output, axis=1, keepdims=True)

        # cache the output so that we can use it at the backward
        # pass when computing the gradient on input
        self.output_cache = output
        return output

    def backward(self, grad_top):
        pass


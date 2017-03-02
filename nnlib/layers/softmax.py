import numpy as np

from nnlib.layers import Layer


class SoftmaxWithCrossEntropy(Layer):
    """A softmax layer with the cross entropy loss on top. The two layers are
    merged to avoid the computation of a full Jacobian matrix (only ground
    truth scores influence the value of the loss function)."""

    def __init__(self):
        self.y_cache = None
        self.probs_cache = None

    def forward(self, X, y):
        # shift the input so that the highest value is
        # zero (improve numerical stability)
        X -= np.max(X, axis=1).reshape((-1, 1))

        probs = np.exp(X)
        probs /= np.sum(probs, axis=1, keepdims=True)

        # cache the class vector and the output of the softmax
        # layer so that we can use them at the backward pass
        # when computing the gradient on input
        self.y_cache = y
        self.probs_cache = probs

        # ground truths are not present during test time so we can't
        # compute the value of the loss function and just return the
        # probabilities instead
        if y is None:
            return probs

        # compute the value of the loss function
        num_examples = X.shape[0]

        loss_i = - np.log(probs[range(num_examples), y])
        loss = np.mean(loss_i)
        return loss

    def backward(self):
        num_examples = self.probs_cache.shape[0]

        d_X = self.probs_cache.copy()
        d_X[range(num_examples), self.y_cache] -= 1
        d_X /= num_examples
        return d_X

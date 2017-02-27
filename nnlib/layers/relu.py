import numpy as np


class ReLU(object):
    """A rectified linear unit layer."""

    def __init__(self):
        self.X_cache = None

    def forward(self, X):
        # cache the input so that we can use it at the
        # backward pass when computing the gradient on input
        self.X_cache = X

        Z = np.maximum(0, X)
        return Z

    def backward(self, grad_top):
        d_X = grad_top
        d_X[self.X_cache < 0] = 0
        return d_X


class LeakyReLU(object):
    """A leaky rectified linear unit layer.

    Parameters
    ----------
    leakiness: float (default=0.01)
        Slope in the negative part, usually between 0 and 1.
    """

    def __init__(self, leakiness=0.01):
        self.leakiness = leakiness
        self.X_cache = None

    def forward(self, X):
        # cache the input so that we can use it at the
        # backward pass when computing the gradient on input
        self.X_cache = X

        Z = X.copy()
        np.putmask(Z, X < 0, self.leakiness * X)
        return Z

    def backward(self, grad_top):
        d_X = grad_top
        d_X[self.X_cache < 0] *= self.leakiness
        return d_X


class PReLU(object):
    """A parametric rectified linear unit layer."""

    def __init__(self):
        self.leakiness = 0

        self.X_cache = None
        self.d_leakiness = None

    def forward(self, X):
        # cache the input so that we can use it at the
        # backward pass when computing the gradient on input
        self.X_cache = X

        Z = X.copy()
        np.putmask(Z, X < 0, self.leakiness * X)
        return Z

    def backward(self, grad_top):
        d_leakiness = self.X_cache.copy()
        d_leakiness[self.X_cache >= 0] = 0
        d_leakiness *= grad_top
        self.d_leakiness = np.sum(d_leakiness)

        d_X = grad_top
        d_X[self.X_cache < 0] *= self.leakiness
        return d_X

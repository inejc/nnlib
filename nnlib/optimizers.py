from abc import ABC, abstractmethod
from collections import namedtuple

# data container for parameters and gradients name pairs (object's named
# attributes) returned by layers whose parameters are backproped into
# (see nnlib.layers.Layer base class for more details)
ParamsGrads = namedtuple('ParamsGrads', ['params', 'grads'])


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, lr):
        self.lr = lr
        self.layers = []

    @abstractmethod
    def register_layer(self, layer):
        pass

    @abstractmethod
    def make_updates(self):
        pass


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Parameters
    ----------
    lr: float >= 0 (default=0.01)
        Learning rate.
    """

    def __init__(self, lr=0.01):
        super().__init__(lr)

    def register_layer(self, layer):
        self.layers.append(layer)

    def make_updates(self):
        for layer in self.layers:
            self.update_layer(layer)

    def update_layer(self, layer):
        params_grads_names = layer.updatable_params_grads_names()

        for param_name, grad_name in params_grads_names:
            param = getattr(layer, param_name)
            grad = getattr(layer, grad_name)

            param += - self.lr * grad
            setattr(layer, param_name, param)

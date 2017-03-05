from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, lr):
        self._lr = lr
        self._layers = []

    @abstractmethod
    def register_layer(self, layer):
        """Should index the layer and init anything needed to update
        it later."""
        pass

    @abstractmethod
    def update_layers(self):
        """Should update all registered (updatable) layers' weights."""
        pass


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Parameters
    ----------
    lr: float >= 0, default 0.01
        Learning rate.
    """

    def __init__(self, lr=0.01):
        super().__init__(lr)

    def register_layer(self, layer):
        self._layers.append(layer)

    def update_layers(self):
        for layer in self._layers:
            self._update_layer(layer)

    def _update_layer(self, layer):
        params_grads_names = layer.updatable_params_grads_names()

        for param_name, grad_name in params_grads_names:
            param = getattr(layer, param_name)
            grad = getattr(layer, grad_name)

            param += - self._lr * grad
            setattr(layer, param_name, param)

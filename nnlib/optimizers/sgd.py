from nnlib.optimizers import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Parameters
    ----------
    lr: float >= 0, default 0.01
        Learning rate.
    """

    def __init__(self, lr=0.01):
        super().__init__(lr)
        self._layers = []

    def register_layer(self, updatable_layer):
        self._layers.append(updatable_layer)

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

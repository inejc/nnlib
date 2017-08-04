from nnlib.optimizers import Optimizer


class SGD(Optimizer):
    """Vanilla stochastic gradient descent optimizer.

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
        params_grads_names = layer.get_updatable_params_grads_names()

        for param_name, grad_name in params_grads_names:
            param = getattr(layer, param_name)
            grad = getattr(layer, grad_name)

            param += - self._lr * grad
            setattr(layer, param_name, param)


class SGDMomentum(Optimizer):
    """Stochastic gradient descent optimizer with momentum updates.

    Parameters
    ----------
    lr: float >= 0, default 0.01
        Learning rate.

    momentum: float > 0 and <= 1, default 0.9
        Hyperparameter that controls the momentum, could be interpreted
        as the coefficient of friction.

    nesterov: bool, default True
        Whether to use the Nesterov momentum update.
    """

    def __init__(self, lr=0.01, momentum=0.9, nesterov=True):
        super().__init__(lr)
        self._layers_caches = []
        self._momentum = momentum
        self._nesterov = nesterov

    def register_layer(self, updatable_layer):
        num_params = len(updatable_layer.get_updatable_params_grads_names())
        layer_cache = {
            'layer': updatable_layer,
            'velocities': [0 for _ in range(num_params)]
        }

        self._layers_caches.append(layer_cache)

    def update_layers(self):
        for layer_cache in self._layers_caches:
            self._update_layer(layer_cache)

    def _update_layer(self, layer_cache):
        layer = layer_cache['layer']
        velocities = layer_cache['velocities']

        params_grads_names = layer.get_updatable_params_grads_names()

        for v_i, param_grad_name in enumerate(params_grads_names):
            param_name, grad_name = param_grad_name
            param = getattr(layer, param_name)
            grad = getattr(layer, grad_name)

            v = velocities[v_i]

            if self._nesterov is True:
                v_prev = v

            # integrate velocity and store it
            v = self._momentum * v - self._lr * grad
            velocities[v_i] = v

            # perform actual parameter update
            if self._nesterov is True:
                param += - self._momentum * v_prev + (1 + self._momentum) * v
            else:
                param += v

            setattr(layer, param_name, param)

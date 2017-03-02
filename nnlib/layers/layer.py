from abc import ABC, abstractmethod


class Layer(ABC):
    """Base class for all layers implementations."""

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass

    @property
    def has_updatable_params(self):
        try:
            self.updatable_params_grads_names()
        except NotImplementedError:
            return False

        return True

    def updatable_params_grads_names(self):
        """Expose all parameters and gradients name pairs (object's
        named attributes) that are backproped into.

        Returns
        -------
            A list of nnlib.optimizers.ParamsGrads namedtuples.
        """
        raise NotImplementedError()

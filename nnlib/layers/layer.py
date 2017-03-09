from abc import ABC, abstractmethod
from collections import namedtuple

# data container for parameters and gradients name pairs (object's named
# attributes) returned by layers whose parameters are backproped into
# (see the updatable_params_grads_names() method of the nnlib.layers.Layer
# base class for more details)
ParamGradNames = namedtuple('ParamGradNames', ['param_name', 'grad_name'])


class Layer(ABC):
    """Base class for all layers implementations."""

    @abstractmethod
    def forward(self, *args):
        """Should perform a local forward pass of the layer. Parameters
        should be the inputs to the layer and the outputs of the layer
        should be returned."""
        pass

    @abstractmethod
    def backward(self, *args):
        """Should perform a local backward pass of the layer. Parameters
        should be the gradients from the top (i.e. how layer's outputs
        influence the loss) and the gradients on inputs to the layer should
        be returned."""
        pass

    def is_regularized(self):
        """Indicates whether the layer has weights that contribute to the
        loss value (i.e. exposes weight penalty)."""
        try:
            self.get_regularization_loss()
        except NotImplementedError:
            return False

        return True

    def get_regularization_loss(self):
        """Should return the value of the regularization loss (i.e. the
        weight penalty)."""
        raise NotImplementedError()

    def has_updatable_params(self):
        """Indicates whether the layer has parameters that should be
        backproped into or not."""
        try:
            self.get_updatable_params_grads_names()
        except NotImplementedError:
            return False

        return True

    def get_updatable_params_grads_names(self):
        """Should expose all parameters and gradients name pairs (object's
        named attributes) that are backproped into. Should return a list
        of nnlib.layers.ParamGradNames namedtuples.
        """
        raise NotImplementedError()

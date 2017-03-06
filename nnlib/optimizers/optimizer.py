from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, lr):
        self._lr = lr

    @abstractmethod
    def register_layer(self, updatable_layer):
        """Should index the layer and init anything needed to update
        it later."""
        pass

    @abstractmethod
    def update_layers(self):
        """Should update all registered (updatable) layers' parameters."""
        pass

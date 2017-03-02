

class Model(object):
    """A computational graph that maintaints the connectivity of the layers."""

    def __init__(self):
        self.layers = []
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer):
        self.optimizer = optimizer

        for layer in self.layers:
            if layer.has_updatable_params:
                optimizer.register_layer(layer)

    def forward(self):
        pass

    def backward(self):
        pass

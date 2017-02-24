

class Model(object):
    """A computational graph that maintaints the connectivity of the layers."""

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self):
        pass

    def backward(self):
        pass

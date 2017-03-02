
LAST_LAYER = -1


class Model(object):
    """A computational graph that maintains the connectivity of the layers."""
    # todo: report on training loss and accuracy

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

    def train(self, X, y, batch_size, num_epochs):
        # todo: batch training
        for epoch in range(num_epochs):
            self.forward(X, y)
            self.backward()
            self.optimizer.make_updates()

    def predict(self, X):
        return self.forward(X)

    def forward(self, X, y=None):
        for layer in self.layers[:LAST_LAYER]:
            X = layer.forward(X)

        # pass ground truths to the last layer during the forward pass
        return self.layers[LAST_LAYER].forward(X, y)

    def backward(self):
        # no gradient from the top at the beginning of the backward pass
        grad_top = self.layers[LAST_LAYER].backward()

        for layer in reversed(self.layers[:LAST_LAYER]):
            grad_top = layer.backward(grad_top)

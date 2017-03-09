import numpy as np

from nnlib.utils import yield_data_in_batches, classification_accuracy

_LAST_LAYER = -1


class Model(object):
    """A computational graph that maintains the connectivity of the layers."""

    def __init__(self):
        self._layers = []
        self._optimizer = None
        self._verbose = False

    def add(self, layer):
        """Adds a new layer to the computational graph. The order of
        the layers should be the same as in the forward pass.

        Parameters
        ----------
        layer: nnlib.layers.Layer
            A new layer added to the model.
        """
        self._layers.append(layer)

    def compile(self, optimizer):
        """Prepares the model for training.

        Parameters
        ----------
        optimizer: nnlib.optimizers.Optimizer
            Optimizer used during the training process.
        """
        self._optimizer = optimizer

        for layer in self._layers:
            if layer.has_updatable_params():
                self._optimizer.register_layer(layer)

    def fit(self, X, y, batch_size, num_epochs, shuffle=True, verbose=False):
        """Trains the model.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data.

        y: array-like, shape (n_samples,)
            The target values.

        batch_size: int
            Number of examples per single gradient update.

        num_epochs: int
            Number of iterations to train the model (i.e. number
            of times every example is seen during training).

        shuffle: bool, default True
            Whether the examples are shuffled or not at each epoch
            before putting them in batches.

        verbose: bool, default False
            Whether to report the training loss and accuracy during
            training.
        """
        self._verbose = verbose

        for epoch_index in range(num_epochs):
            self._train_one_epoch(X, y, batch_size, shuffle, epoch_index)

    def _train_one_epoch(self, X, y, batch_size, shuffle, epoch_index):
        """Trains the model for one epoch. For parameters see the
        train() method."""
        X_y_batches = yield_data_in_batches(
            X=X, y=y,
            batch_size=batch_size,
            shuffle=shuffle
        )

        for X_batch, y_batch in X_y_batches:
            self._forward(X_batch, y_batch)
            self._backward()
            self._optimizer.update_layers()

        if self._verbose:
            self._report_after_epoch(X, y, epoch_index)

    def _report_after_epoch(self, X, y, epoch_index):  # pragma: no cover
        """Logs the loss and classification accuracy of the training
        set to stdout."""
        training_loss = self._forward(X, y)
        y_pred = self.predict(X)
        training_acc = classification_accuracy(y, y_pred)

        report = "EPOCH {:d}: training loss: {:f} ~~~ training acc: {:f}"
        report = report.format(epoch_index, training_loss, training_acc)
        print(report)

    def predict(self, X):
        """Makes predictions and returns the predicted classes.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """Makes predictions and returns the probabilities.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input data.
        """
        return self._forward(X)

    def _forward(self, X, y=None):
        """Performs a complete forward pass on all layers in the
        computational graph."""
        regularization_loss = 0

        for layer in self._layers[:_LAST_LAYER]:
            X = layer.forward(X)

            if layer.is_regularized():
                regularization_loss += layer.get_regularization_loss()

        if y is None:
            # probabilities are returned from the last layer if
            # ground truths are not passed to the forward function
            return self._layers[_LAST_LAYER].forward(X)

        loss = self._layers[_LAST_LAYER].forward(X, y)
        return loss + regularization_loss

    def _backward(self):
        """Performs a complete backward pass on all layers in the
        computational graph."""
        # no gradient from the top at the beginning of the backward pass
        grad_top = self._layers[_LAST_LAYER].backward()

        for layer in reversed(self._layers[:_LAST_LAYER]):
            grad_top = layer.backward(grad_top)

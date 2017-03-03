from itertools import islice

import numpy as np
from numpy.random import permutation


def numerical_grad(func, input_, h=1e-6):
    """Computes partial derivatives of func wrt. input_ using the
    center divided difference method. Used to gradient check
    analytical solutions.

    Parameters
    ----------
    func: callable
        A function whose derivatives should be computed.

    input_: scalar or array-like
        Partial derivatives are computed wrt. input_.

    h: float, default 1e-6
        A spacing used when computing the difference, should
        be small.

    Returns
    -------
        grad: scalar or array-like of shape input_.shape
    """

    if np.isscalar(input_):
        return np.sum((func(input_ + h) - func(input_ - h)) / (2 * h))

    grad = np.zeros(input_.shape)

    for i in np.ndindex(input_.shape):
        forward = input_.copy()
        forward[i] += h

        backward = input_.copy()
        backward[i] -= h

        center_divided_diff = (func(forward) - func(backward)) / (2 * h)
        grad[i] = np.sum(center_divided_diff)

    return grad


def yield_data_in_batches(batch_size, X, y=None, shuffle=True):
    """Generates batches of input data.

    Parameters
    ----------
    batch_size: int
        Number of examples in a single batch.

    X: array-like, shape (n_samples, n_features)
        The input data.

    y: array-like, shape (n_samples,)
        The target values. Can be omitted.

    shuffle: bool, default True
        Whether the examples are shuffled or not before
        put into batches.
    """
    num_rows = X.shape[0]

    if shuffle:
        indices_gen = (i for i in permutation(num_rows))
    else:
        indices_gen = (i for i in np.arange(num_rows))

    num_yielded = 0

    while True:
        batch_indices = list(islice(indices_gen, batch_size))
        num_yielded += len(batch_indices)

        if y is None:
            yield X[batch_indices]
        else:
            yield X[batch_indices], y[batch_indices]

        if num_yielded == num_rows:
            return


def classification_accuracy(y, y_pred):
    """Computes the classification accuracy.

    Parameters
    ----------
    y: array-like, shape (n_samples,)
        The true target values (i.e. the ground truths).

    y_pred: array-like, shape (n_samples,)
        The predicted target values.
    """
    return np.mean(np.equal(y, y_pred))

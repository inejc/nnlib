from collections import namedtuple

import numpy as np

# data container for regularization loss and gradient functions
Regularizer = namedtuple('Regularizer', ['loss', 'grad'])


def l2(lambda_):
    """Constructs an L2 nnlib.regularizers.Regularizer namedtuple.

    lambda_: float >= 0
        Regularization strength. Zero equals no regularization,
        higher values mean stronger regularization.
    """
    return Regularizer(
        loss=lambda W: _l2(W, lambda_),
        grad=lambda W: _d_l2(W, lambda_)
    )


def _l2(W, lambda_):
    """Computes the L2 (ridge) regularization loss (i.e. weights penalty
    for the W weights)."""
    # the 0.5 constant simplifies the gradient expression below
    # (the _d_l2 function)
    return 0.5 * lambda_ * np.sum(W * W)


def _d_l2(W, lambda_):
    """Computes the L2 (ridge) regularization gradient on the W weights."""
    return lambda_ * W

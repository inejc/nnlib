import numpy as np


def numerical_grad(func, input_, h=1e-6):
    """Computes partial derivatives of func wrt. input_
    using the center divided difference method."""
    grad = np.zeros(input_.shape)

    for i in np.ndindex(input_.shape):
        forward = input_.copy()
        forward[i] += h

        backward = input_.copy()
        backward[i] -= h

        center_divided_diff = (func(forward) - func(backward)) / (2 * h)
        grad[i] = np.sum(center_divided_diff)

    return grad

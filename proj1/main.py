import unittest

import numpy as np

from activations import Activation
from losses import Loss
from network import Layer, MLPNetwork

if __name__ == '__main__':
    model = MLPNetwork(
        5,
        [
            Layer(10, Activation.relu, Activation.Gradient.relu),
            Layer(10, Activation.relu, Activation.Gradient.relu),
            Layer(2, Activation.sigmoid, Activation.Gradient.sigmoid),
        ],
        Loss.mse,
        Loss.Gradient.mse,
        True,
        42
    )

    X = np.random.uniform(-10, 10, (1, 5))
    y = np.random.uniform(-10, 10, (1, 2))
    model._forward(X, cached_activations=True)
    print(model._backward(y))
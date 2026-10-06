import itertools
from collections.abc import Callable

import numpy as np


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class MLPNetwork:
    def __init__(
            self, 
            input_size: int,
            layers: list[tuple[int, Callable[[np.ndarray], np.ndarray]]],
            include_bias: bool, 
            seed: int
    ) -> None:
        rnd = np.random.default_rng(seed=seed)
        layer_sizes = [input_size] + [layer[0] for layer in layers]
        self.weights = [rnd.normal(0, 2/in_size, (in_size, out_size)) # He initialization
                       for in_size, out_size in itertools.pairwise(layer_sizes)]
        self.biases = [np.zeros(out_size) for out_size in layer_sizes[1:]]
        
        self.activations = [layer[1] for layer in layers]
        self.include_bias = include_bias
        self.seed = seed

    def predict(self, x: np.ndarray) -> np.ndarray:
        for weights, bias, activation in zip(self.weights, self.biases, self.activations):
            x = activation(x @ weights + bias)
        return x
    
if __name__ == '__main__':
    model = MLPNetwork(
        10,
        [
            (5, relu),
            (20, sigmoid),
            (30, sigmoid),
            (5, relu),
        ],
        include_bias=True,
        seed=42
    )

    print(model.predict(np.random.uniform(-10, 10, 10)))
import itertools
from enum import Enum, auto

import numpy as np
from numpy.typing import ArrayLike


class ActivationType(Enum):
    RELU = auto()
    SIGMOID = auto()

ACTIVATION_FN = {
    ActivationType.RELU: lambda x: np.maximum(0, x),
    ActivationType.SIGMOID: lambda x: 1/(1 + np.exp(-x)),
}

ACTIVATION_GRAD = {
    ActivationType.RELU: lambda x: np.where(x <= 0, 0, 1),
    ActivationType.SIGMOID: lambda x: np.exp(-x)/((1-np.exp(-x))**2),
}

class LossType(Enum):
    MSE = auto()

LOSS_FN = {
    LossType.MSE: lambda y, y_target: np.sum((y_target-y)**2)/2,
}

LOSS_GRAD = {
    LossType.MSE: lambda y, y_target: y-y_target,
}


class MLPNetwork:
    def __init__(
            self, 
            input_size: int,
            layers: list[tuple[int, ActivationType]],
            loss: LossType,
            include_bias: bool, 
            seed: int
    ) -> None:
        rnd = np.random.default_rng(seed=seed)
        layer_sizes = [input_size] + [layer[0] for layer in layers]
        self.weights = [rnd.normal(0, 2/in_size, (in_size, out_size)) # He initialization
                       for in_size, out_size in itertools.pairwise(layer_sizes)]
        self.biases = [np.zeros(out_size) for out_size in layer_sizes[1:]]
        
        self.activation_funs = [layer[1] for layer in layers]
        
        self.loss = loss
        self.include_bias = include_bias
        self.seed = seed

    def predict(self, x: np.ndarray) -> np.ndarray:
        for weights, bias, activation in zip(self.weights, self.biases, self.activation_funs):
            x = ACTIVATION_FN[activation](x @ weights + bias)
        return x
    
    def backpropagate(self, x: np.ndarray, y_target: ArrayLike) -> tuple[list[np.ndarray], list[np.ndarray]]:
        x = np.atleast_2d(x)
        y_target = np.atleast_2d(y_target)
        
        zs = []
        activations = []
        for weights, bias, activation in zip(self.weights, self.biases, self.activation_funs):
            activations.append(x)
            z = x @ weights + bias
            zs.append(z)
            x = ACTIVATION_FN[activation](z)
        activations.append(x)
    
        delta = LOSS_GRAD[self.loss](activations[-1], y_target) * \
            ACTIVATION_GRAD[self.activation_funs[-1]](zs[-1])

        grad_bias = [delta]
        grad_weights = [activations[-2].T @ delta]
        
        for i in range(-2, -len(zs)-1, -1):
            act_grad = ACTIVATION_GRAD[self.activation_funs[i]](zs[i])
            delta = (delta @ self.weights[i+1].T) * act_grad
            grad_bias.append(delta)
            grad_weights.append(activations[i-1].T @ delta)

        return grad_bias[::-1], grad_weights[::-1]
    
if __name__ == '__main__':
    model = MLPNetwork(
        1,
        [
            (5, ActivationType.RELU),
            (1, ActivationType.SIGMOID),
        ],
        loss=LossType.MSE,
        include_bias=True,
        seed=42
    )

    x = np.random.uniform(-10, 10, 1)
    print(x)
    print(model.predict(x))
    print(model.backpropagate(x, 10))
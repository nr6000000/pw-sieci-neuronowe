import itertools
from collections.abc import Callable
from activations import *
from losses import *

import numpy as np


class MLPNetwork:
    def __init__(
            self,
            input_size: int,
            layers: list[tuple[int, Callable[[np.ndarray], np.ndarray]]],
            loss_function: Callable[[np.ndarray, np.ndarray], float],
            include_bias: bool,
            seed: int
    ) -> None:
        self.loss_history: list[float] = []
        self.weight_history: list[float] = []
        self.cached_before_activations: list[np.ndarray] = []
        self.cached_after_activations: list[np.ndarray] = []

        self.seed = seed
        self.include_bias = include_bias
        self.loss_function = loss_function

        rng = np.random.default_rng(seed=seed)
        self.activation_functions = [layer[1] for layer in layers]

        layer_sizes = [input_size] + [layer[0] for layer in layers]
        self.weights = [rng.normal(0, np.sqrt(2 / in_size), (in_size, out_size)) # He initialization
                       for in_size, out_size in itertools.pairwise(layer_sizes)]
        self.biases = [np.zeros(out_size) for out_size in layer_sizes[1:]]

    def _forward(self, X: np.ndarray, cached_activations=False) -> np.ndarray:
        if cached_activations:
            self.cached_before_activations = []
            # needed for 
            self.cached_after_activations = [X]
        for weights, bias, activation in zip(self.weights, self.biases, self.activations):
            X = X @ weights + bias if self.include_bias else np.zeros(shape=X.shape)
            if cached_activations:
                self.cached_before_activations.append(X)
            X = activation(X)
            if cached_activations:
                self.cached_after_activations.append(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        # only predict without caching for backpropagation
        return self._forward(X)

    def _backward(self, y_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        activations = self.cached_after_activations
        zs = self.cached_before_activations
        n = X.shape[0]

        delta = self.loss_function(activations[-1], y_target) * \
            self.activation_functions[-1](zs[-1])

        grad_biases = [np.sum(delta, axis=0, keepdims=True) / n]
        grad_weights = [activations[-2].T @ delta / n]
        
        for i in range(-2, -len(zs)-1, -1):
            act_grad = self.activation_functions[i](zs[i])
            delta = (delta @ self.weights[i+1].T) * act_grad
            grad_biases.append(np.sum(delta, axis=0, keepdims=True) / n)
            grad_weights.append(activations[i-1].T @ delta / n)

        return grad_weights[::-1], grad_biases[::-1]

    def _update_weights(self, grad_weights: np.ndarray, grad_biases: np.ndarray, learning_rage: float) -> None:
        for self_weights, in_weights in zip(self.weights, grad_weights):
            self_weights -= in_weights * learning_rage + grad_biases if self.include_bias else np.zeros(shape=self_weights.shape)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int=1000, learning_rate: float=0.015, batch_size: int=64, history_saved=False) -> None:
        if history_saved:
            self.loss_history = []
            self.weight_history = []

        batches_num = len(X)

        for epoch in epochs:
            shuffled_indices = self.rng.permutation(batches_num)
            for batch_index in range(0, batches_num, batch_size):
                batch_indices = indices[batch_]

                X_batch, y_batch = X[batch_indices], y[batch_indices]
                y_pred = self._forward(X_batch, cached_activations=True)
                loss = self.loss_function

                grad_weights, grad_biases = self._backward()

                self._update_weights(grad_weights, grad_biases, learning_rate)

                if history_saved:
                    self.loss_history.append(loss)
                    self.weight_history.append()


import itertools
from collections.abc import Callable

import numpy as np
from activations import Activation
from losses import Loss


class Layer:
    def __init__(
        self,
        neurons: int,
        activation_function: Callable[[np.ndarray], np.ndarray],
        activation_gradient_function: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.neurons = neurons
        self.activation_function = activation_function
        self.activation_gradient_function = activation_gradient_function


class MLPNetwork:
    def __init__(
        self,
        input_size: int,
        layers: list[Layer],
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        loss_gradient_function: Callable[[np.ndarray, np.ndarray], float],
        include_bias: bool,
        seed: int | None = None,
    ) -> None:
        self.loss_history: list[float] = []
        self.weight_history: list[list[np.ndarray]] = []
        self.bias_history: list[list[np.ndarray]] = []
        self.cached_zs: list[np.ndarray] = []
        self.cached_activations: list[np.ndarray] = []

        self.seed = seed
        self.include_bias = include_bias
        self.loss_fn = loss_function
        self.loss_grad = loss_gradient_function

        self.rng = np.random.default_rng(seed=seed)
        self.activation_fns = [layer.activation_function for layer in layers]
        self.activation_grads = [layer.activation_gradient_function for layer in layers]

        layer_sizes = [input_size] + [layer.neurons for layer in layers]
        self.weights = [
            self.rng.normal(0, np.sqrt(2 / in_size), (in_size, out_size))  # He initialization
            for in_size, out_size in itertools.pairwise(layer_sizes)
        ]
        self.biases = [np.zeros(out_size) for out_size in layer_sizes[1:]]

    def _forward(self, X: np.ndarray, cached_activations=False) -> np.ndarray:
        if cached_activations:
            self.cached_zs = []
            self.cached_activations = [X]
        for weights, biases, activation in zip(self.weights, self.biases, self.activation_fns):
            X = X @ weights
            if self.include_bias:
                X += biases
            if cached_activations:
                self.cached_zs.append(X)
            X = activation(X)
            if cached_activations:
                self.cached_activations.append(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        # only predict without caching for backpropagation
        return self._forward(X)

    def _backward(self, y_target: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = self.cached_activations
        zs = self.cached_zs
        n = y_target.shape[0]

        # Delta computation specialization
        if (self.loss_fn is Loss.bce and self.activation_fns[-1] is Activation.sigmoid) or (
            self.loss_fn is Loss.cce and self.activation_fns[-1] is Activation.softmax
        ):
            delta = activations[-1] - y_target
        elif self.activation_fns[-1] is Activation.softmax:
            a = activations[-1]
            grad_a = self.loss_grad(y_target, a)
            delta = a * (grad_a - grad_a @ a.T)
        else:
            delta = self.loss_grad(y_target, activations[-1]) * self.activation_grads[-1](zs[-1])

        grad_biases = [np.sum(delta, axis=0) / n]
        grad_weights = [activations[-2].T @ delta / n]

        for i in range(-2, -len(zs) - 1, -1):
            act_grad = self.activation_grads[i](zs[i])
            delta = (delta @ self.weights[i + 1].T) * act_grad
            grad_biases.append(np.sum(delta, axis=0) / n)
            grad_weights.append(activations[i - 1].T @ delta / n)

        return grad_weights[::-1], grad_biases[::-1]

    def _update_parameters(
        self, grad_weights: list[np.ndarray], grad_biases: list[np.ndarray], learning_rage: float
    ) -> None:
        for i in range(len(self.weights)):
            self.weights[i] -= grad_weights[i] * learning_rage
            if self.include_bias:
                self.biases[i] -= grad_biases[i] * learning_rage

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.015,
        batch_size: int = 64,
        history_saved=False,
    ) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must be 2D")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Differing number of samples between X and y")

        if X.shape[1] != self.weights[0].shape[0]:
            raise ValueError("X.shape[1] != self.weights[0].shape[0]")

        if y.shape[1] != self.weights[-1].shape[1]:
            raise ValueError("y.shape[1] != self.weights[-1].shape[1]")

        if history_saved:
            self.loss_history = []
            self.weight_history = []
            self.bias_history = []

        batches_num = len(X)

        for _ in range(epochs):
            shuffled_indices = self.rng.permutation(batches_num)
            for batch_start in range(0, batches_num, batch_size):
                batch_indices = shuffled_indices[batch_start : batch_start + batch_size]

                X_batch, y_batch = X[batch_indices], y[batch_indices]
                y_pred = self._forward(X_batch, cached_activations=True)
                loss = self.loss_fn(y_batch, y_pred)

                if history_saved:
                    self.loss_history.append(loss)
<<<<<<< HEAD
                    self.weight_history.append([w.copy() for w in self.weights])
                    if self.include_bias:
                        self.bias_history.append([b.copy() for b in self.biases])
=======
                    # TODO: Full weight snapshots per batch may exhaust memory on MNI
                    self.weight_history.append([w.copy() for w in self.weights])
>>>>>>> f72b50f (Add experiments and pre-commit YAML)

                grad_weights, grad_biases = self._backward(y_batch)

                self._update_parameters(grad_weights, grad_biases, learning_rate)

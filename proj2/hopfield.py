import os
os.environ["NUMBA_CPU_NAME"] = "generic"

from typing import Literal

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def async_sweep(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = v.copy()
    for i in np.random.permutation(v.shape[0]):
        h = W[i] @ v
        if h >= 0:
            v[i] = 1
        else:
            v[i] = -1
    return v

class HopfieldNet:
    def __init__(
            self, 
            synchronous: bool,
            update_rule: Literal['hebb', 'oja'],
            max_steps: int,
            learning_rate: float,
            epochs: int,
    ) -> None:
        self.synchronous = synchronous
        self.max_steps = max_steps
        self.update_rule = update_rule
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n_samples, size = X.shape

        if self.update_rule == 'hebb':
            # for i in range(X.shape[1]):
            #     for j in range(i+1, X.shape[1]):
            #         weight = np.mean(X[:,i] * X[:,j])
            #         self.weights[i,j] = weight
            #         self.weights[j,i] = weight

            self.weights = (X.T @ X) / n_samples
            np.fill_diagonal(self.weights, 0)
        elif self.update_rule == 'oja':
            # Inna opcja, bez broadcastu, chyba bardziej zgodne
            # z teoretycznym Oją bo y wyliczany jest po zmodyfikowanej wadze 
            # for i in range(X.shape[1]):
            #     y = self.weights[i,:] @ X[k]
            #     delta_weight = self.learning_rate*y*(X[k] - y*self.weights[i,:])
            #     self.weights[i,:] += delta_weight
            #     self.weights[:,i] += delta_weight

            W = np.random.normal(0, 1/np.sqrt(size), (size, size))
            W = (W + W.T) / 2
            np.fill_diagonal(W, 0)

            for _ in range(self.epochs):
                for k in np.random.permutation(n_samples):
                    y = W @ X[k]
                    delta = self.learning_rate * (np.outer(y, X[k]) - (y ** 2)[:, None] * W)
                    W += (delta + delta.T) / 2
                    np.fill_diagonal(W, 0)
            self.weights = W
        else:
            raise ValueError(f"Unknown update rule: {self.update_rule}")
    
    def predict(self, X: np.ndarray, save_steps: bool = False):
        steps = 0
        saved_steps = []

        out = X.copy()
        while steps < self.max_steps:
            if save_steps:
                saved_steps.append(out.copy())

            if self.synchronous:
                new_out = self.weights @ out
                new_out = np.where(new_out >= 0, 1, -1)
            else:
                new_out = async_sweep(self.weights, out)

            if np.array_equal(new_out, out): 
                break

            out = new_out
            steps += 1

        if save_steps:
            return saved_steps
        else:
            return out
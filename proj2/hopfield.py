import os
from typing import Literal

os.environ["NUMBA_CPU_NAME"] = "generic"

import numba as nb
import numpy as np

ZERO_FIELD_ATOL = 1e-12


@nb.jit(nopython=True)
def async_sweep(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = v.copy()
    for i in np.random.permutation(v.shape[0]):
        h = W[i] @ v
        if h > ZERO_FIELD_ATOL:
            v[i] = 1
        elif h < -ZERO_FIELD_ATOL:
            v[i] = -1
    return v


class HopfieldNet:
    def __init__(
        self,
        synchronous: bool = True,
        update_rule: Literal["hebb", "oja"] = "hebb",
        max_steps: int = 100,
        learning_rate: float = 0.001,
        epochs: int = 100,
    ) -> None:
        self.synchronous = synchronous
        self.max_steps = max_steps
        self.update_rule = update_rule
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.size = 0
        self.last_status = "not_run"
        self.last_steps = 0

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n_samples, size = X.shape
        if not X.size or size > 5000:
            raise ValueError("Dense training requires 1-5000 neurons and a nonempty pattern set.")

        if self.update_rule == "hebb":
            self.weights = (X.T @ X) / size
            np.fill_diagonal(self.weights, 0)
        elif self.update_rule == "oja":
            W = np.random.normal(0, 1 / np.sqrt(size), (size, size))
            W = (W + W.T) / 2
            np.fill_diagonal(W, 0)

            with np.errstate(over="raise", invalid="raise"):
                for _ in range(self.epochs):
                    for k in np.random.permutation(n_samples):
                        y = W @ X[k]
                        delta = self.learning_rate * (np.outer(y, X[k]) - (y**2)[:, None] * W)
                        W += (delta + delta.T) / 2
                        np.fill_diagonal(W, 0)
            self.weights = W
        else:
            raise ValueError(f"Unknown update rule: {self.update_rule}")
        self.size = size
        self.last_status, self.last_steps = "not_run", 0
        return self

    def field(self, v):
        h = self.weights @ v
        # Roundoff must not flip neurons whose field should be zero.
        h[np.abs(h) <= ZERO_FIELD_ATOL] = 0
        return h

    def _async_sweep(self, v):
        return async_sweep(self.weights, v)

    def energy(self, v):
        return float(-0.5 * v @ self.field(v))

    def predict(self, X: np.ndarray, save_steps: bool = False):
        if not self.size:
            raise RuntimeError("Call fit() before predict().")
        out = np.asarray(X, dtype=float).copy()
        if out.shape != (self.size,):
            raise ValueError("Input size does not match the network.")

        saved_steps = [out.copy()] if save_steps else []
        previous = None
        self.last_status, self.last_steps = "max_steps", 0
        for _ in range(self.max_steps):
            if self.synchronous:
                h = self.field(out)
                new_out = np.where(h > 0, 1.0, np.where(h < 0, -1.0, out))
            else:
                new_out = self._async_sweep(out)

            fixed = np.array_equal(new_out, out)
            cycle = self.synchronous and previous is not None and np.array_equal(new_out, previous)
            previous, out = out, new_out
            self.last_steps += 1
            if save_steps:
                saved_steps.append(out.copy())
            if fixed or cycle:
                self.last_status = "fixed_point" if fixed else "cycle_2"
                break

        return saved_steps if save_steps else out

import os
from time import perf_counter
from typing import Literal

os.environ["NUMBA_CPU_NAME"] = "generic"

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
    """Bipolar associative memory.

    The Oja option is the clamped autoassociative variant: during training,
    presynaptic activity and the postsynaptic target both equal the pattern x.
    Delta W = eta * (outer(x, x) - (x**2)[:, None] * W).
    For bipolar x this reduces to eta * (outer(x, x) - W), an exponentially
    weighted correlation estimate. It bounds individual weights, not row norms.
    This is not the autonomous linear PCA neuron.
    The separate "oja_subspace" option learns a principal subspace by batch
    Oja updates with QR retraction, then constructs a projection memory.
    Its budget is max_fit_steps, not the clamped variant's epochs.

    Reference: Baldi and Sadowski (2016), Appendix C, equations 85-86:
    https://arxiv.org/pdf/1506.06472
    """

    OJA_VARIANT = "clamped_autoassociative"
    OJA_SUBSPACE_VARIANT = "batch_oja_subspace_qr_projection"

    def __init__(
        self,
        synchronous: bool,
        update_rule: Literal["hebb", "oja", "oja_subspace"],
        max_steps: int,
        learning_rate: float,
        epochs: int,
        *,
        max_fit_steps: int = 1_000_000,
        fit_tol: float = 1e-6,
        fit_check_every: int = 100,
        random_state: int | None = 42,
    ) -> None:
        self.synchronous = synchronous
        self.max_steps = max_steps
        self.update_rule = update_rule
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_fit_steps = max_fit_steps
        self.fit_tol = fit_tol
        self.fit_check_every = fit_check_every
        self.random_state = random_state
        self.basis_: np.ndarray | None = None
        self.fit_info: dict = {}
        self.last_status = "not_run"
        self.last_steps = 0
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        started = perf_counter()
        self.weights = None
        self.basis_ = None
        self.fit_info = {}
        self.last_status = "not_run"
        self.last_steps = 0
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or 0 in X.shape:
            raise ValueError("Expected a nonempty matrix of training patterns.")
        if not np.all((X == -1) | (X == 1)):
            raise ValueError("Training patterns must contain only -1 and +1.")

        n_samples, size = X.shape
        self.fit_info = {
            "algorithm_variant": self.update_rule,
            "training_units": None,
            "fit_steps": 0,
            "data_passes": None,
            "converged": None,
            "residual": None,
            "n_components": None,
            "training_rank": None,
            "initialization_seed": None,
        }
        if self.update_rule == "hebb":
            self.fit_info.update(
                algorithm_variant="hebb_correlation",
                training_units="closed_form",
                data_passes=1,
            )
            W = (X.T @ X) / n_samples
            np.fill_diagonal(W, 0)
        elif self.update_rule == "oja":
            eta = float(self.learning_rate)
            if not np.isfinite(eta) or not 0 < eta <= 1:
                raise ValueError("Clamped Oja requires 0 < learning_rate <= 1.")
            if not isinstance(self.epochs, (int, np.integer)) or self.epochs < 1:
                raise ValueError("epochs must be a positive integer.")

            # Zero initialization adds no arbitrary associations.
            # With clamped bipolar activities, each update preserves symmetry
            # and keeps all weights in [-1, 1] for 0 < eta <= 1.
            self.fit_info.update(
                algorithm_variant=self.OJA_VARIANT,
                training_units="online_updates",
                fit_steps=self.epochs * n_samples,
                data_passes=self.epochs,
            )
            W = np.zeros((size, size), dtype=np.float64)
            with np.errstate(over="raise", invalid="raise"):
                for _ in range(self.epochs):
                    for k in np.random.permutation(n_samples):
                        x = X[k]
                        W += eta * (np.outer(x, x) - W)
                        np.fill_diagonal(W, 0)
        elif self.update_rule == "oja_subspace":
            W = self._fit_oja_subspace(X)
        else:
            raise ValueError(f"Unknown update rule: {self.update_rule}")

        if not np.isfinite(W).all():
            raise FloatingPointError("Training produced NaN or infinite weights.")
        self.weights = W
        self.fit_info["fit_seconds"] = perf_counter() - started

    def _fit_oja_subspace(self, X: np.ndarray) -> np.ndarray:
        """Learn an uncentered subspace from a random basis, then embed it.

        C = X.T @ X / P; Delta B = eta * (C B - B (B.T C B)).
        Each full-batch update is followed by QR retraction. Only the rank
        (dimension) is obtained from matrix_rank; data singular vectors or
        a pseudoinverse are never used to construct the learned basis.

        W = B @ B.T with the diagonal removed is the recurrent memory.
        Reconstruction residual measures subspace convergence, separately
        from fixed-point stability and retrieval. A cap is not convergence.
        See Oja (1989), Neural Networks, Principal Components, and Subspaces;
        Bonnabel (2013), section 4.1, equation 14 (QR retraction):
        https://arxiv.org/pdf/1111.5280
        """
        eta = float(self.learning_rate)
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        for name, value in (
            ("max_fit_steps", self.max_fit_steps),
            ("fit_check_every", self.fit_check_every),
        ):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        tol = float(self.fit_tol)
        if not np.isfinite(tol) or not 0 < tol < 1:
            raise ValueError("fit_tol must lie strictly between 0 and 1.")

        p, n = X.shape
        rank = int(np.linalg.matrix_rank(X))
        if rank == n:
            raise ValueError(
                "Full-rank data gives the identity projector; removing its "
                "diagonal leaves no associative connections."
            )

        # Used only to bound the learning step, never as learned weights.
        max_eigenvalue = float(np.linalg.eigvalsh((X @ X.T) / p)[-1])
        if eta * max_eigenvalue > 1 + 1e-12:
            raise ValueError(
                "Subspace learning requires learning_rate <= "
                f"{1 / max_eigenvalue:.6g} for these data."
            )

        rng = np.random.default_rng(self.random_state)
        B = np.linalg.qr(rng.normal(size=(n, rank)), mode="reduced")[0]
        x_norm = np.linalg.norm(X, ord="fro")

        def residual():
            return float(np.linalg.norm(X - (X @ B) @ B.T, ord="fro") / x_norm)

        relative_residual = residual()
        step = 0
        if relative_residual > tol:
            with np.errstate(over="raise", invalid="raise"):
                for step in range(1, self.max_fit_steps + 1):
                    CB = X.T @ (X @ B) / p
                    B += eta * (CB - B @ (B.T @ CB))
                    B = np.linalg.qr(B, mode="reduced")[0]
                    if not np.isfinite(B).all():
                        raise FloatingPointError("Non-finite Oja subspace basis.")
                    if step % self.fit_check_every == 0 or step == self.max_fit_steps:
                        relative_residual = residual()
                        if relative_residual <= tol:
                            break

        W = B @ B.T
        np.fill_diagonal(W, 0)
        self.basis_ = B
        self.fit_info.update(
            algorithm_variant=self.OJA_SUBSPACE_VARIANT,
            training_units="batch_updates",
            fit_steps=step,
            data_passes=step,
            converged=bool(relative_residual <= tol),
            residual=relative_residual,
            n_components=rank,
            training_rank=rank,
            initialization_seed=(int(self.random_state) if self.random_state is not None else None),
            max_fit_steps=self.max_fit_steps,
            fit_tol=tol,
            check_every=self.fit_check_every,
            learning_rate=eta,
            covariance_max_eigenvalue=max_eigenvalue,
            centered=False,
            qr_retraction=True,
        )
        return W

    def predict(self, X: np.ndarray, save_steps: bool = False):
        if self.weights is None:
            raise RuntimeError("Call fit() successfully before predict().")
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("Weights contain NaN or infinity.")

        out = np.asarray(X, dtype=np.float64).copy()
        if out.ndim != 1 or out.size != self.weights.shape[0]:
            raise ValueError("Input size does not match the trained network.")
        if not np.isfinite(out).all():
            raise ValueError("Input contains NaN or infinity.")

        saved_steps = [out.copy()] if save_steps else []
        previous = None
        self.last_status = "max_steps"
        self.last_steps = 0

        for _ in range(self.max_steps):
            if self.synchronous:
                new_out = self.weights @ out
                new_out = np.where(new_out >= 0, 1.0, -1.0)
            else:
                new_out = async_sweep(self.weights, out)

            is_fixed_point = np.array_equal(new_out, out)
            is_two_cycle = (
                self.synchronous and previous is not None and np.array_equal(new_out, previous)
            )

            previous, out = out, new_out
            self.last_steps += 1
            if save_steps:
                saved_steps.append(out.copy())

            if is_fixed_point:
                self.last_status = "fixed_point"
                break
            if is_two_cycle:
                self.last_status = "cycle_2"
                break

        return saved_steps if save_steps else out

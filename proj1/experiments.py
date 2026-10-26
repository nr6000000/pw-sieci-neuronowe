from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from activations import Activation
from losses import Loss
from network import Layer, MLPNetwork

ActivationFn = Callable[[np.ndarray], np.ndarray]
LossFn = Callable[[np.ndarray, np.ndarray], float]
LossGradientFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class Result:
    config: dict[str, object]
    train_metric: float
    validation_metric: float
    loss_history: list[float]


ACTIVATIONS: dict[str, tuple[ActivationFn, ActivationFn]] = {
    "sigmoid": (Activation.sigmoid, Activation.Gradient.sigmoid),
    "tanh": (Activation.tanh, Activation.Gradient.tanh),
    "relu": (Activation.relu, Activation.Gradient.relu),
}

LOSSES: dict[str, tuple[LossFn, LossGradientFn]] = {
    "mse": (Loss.mse, Loss.Gradient.mse),
    "mae": (Loss.mae, Loss.Gradient.mae),
    "bce": (Loss.bce, Loss.Gradient.bce),
    "cce": (Loss.cce, Loss.Gradient.cce),
}


class Experiments:
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task: Literal["regression", "classification"] = "regression",
        seed: int = 42,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        batch_size: int = 32,
    ) -> None:
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.task = task
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _metric(self, y: np.ndarray, pred: np.ndarray) -> float:
        if self.task == "regression":
            return float(Loss.mse(y, pred))

        if y.shape[1] == 1:
            return float(np.mean((pred >= 0.5) == y))

        return float(np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1)))

    def _run(
        self,
        hidden: list[int],
        activation: str = "tanh",
        loss: str | None = None,
    ) -> Result:
        if self.task == "regression":
            output = Layer(1, Activation.linear, Activation.Gradient.linear)
            loss = loss or "mse"
        elif self.y_train.shape[1] == 1:
            output = Layer(1, Activation.sigmoid, Activation.Gradient.sigmoid)
            loss = loss or "bce"
        else:
            output = Layer(self.y_train.shape[1], Activation.softmax)
            loss = loss or "cce"

        fn, grad = ACTIVATIONS[activation]
        loss_fn, loss_grad = LOSSES[loss]

        layers = [Layer(n, fn, grad) for n in hidden] + [output]

        model = MLPNetwork(
            input_size=self.X_train.shape[1],
            layers=layers,
            loss_function=loss_fn,
            loss_gradient_function=loss_grad,
            include_bias=True,
            seed=self.seed,
        )

        model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            history_saved=True,
        )

        return Result(
            config={
                "hidden": hidden,
                "activation": activation,
                "loss": loss,
                "seed": self.seed,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
            },
            train_metric=self._metric(self.y_train, model.predict(self.X_train)),
            validation_metric=self._metric(self.y_val, model.predict(self.X_val)),
            loss_history=model.loss_history.copy(),
        )

    def varying_activation(self) -> list[Result]:
        return [self._run([16, 16], activation=a) for a in ("sigmoid", "tanh", "relu")]

    def varying_hidden_layers(self) -> list[Result]:
        return [self._run([16] * depth) for depth in range(5)]

    def varying_neurons(self) -> list[Result]:
        return [self._run([width, width]) for width in (4, 8, 16, 32)]

    def varying_regression_loss(self) -> list[Result]:
        if self.task != "regression":
            raise ValueError("Regression task required")

        return [self._run([16, 16], loss=loss) for loss in ("mse", "mae")]

    def varying_classification_loss(self) -> list[Result]:
        if self.task != "classification" or self.y_train.shape[1] != 1:
            raise ValueError("Binary classification required")

        return [self._run([16, 16], loss=loss) for loss in ("bce", "mse")]

    def run(self) -> None:
        pass

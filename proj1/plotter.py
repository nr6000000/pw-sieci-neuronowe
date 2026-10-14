import numpy as np


class Plotter:
    @staticmethod
    def plot_losses(loss_history: list[float]) -> None:
        pass

    @staticmethod
    def plot_weights(weight_history: list[float]) -> None:
        pass

    @staticmethod
    def plot_classification_results(X: np.ndarray, y: np.ndarray) -> None:
        pass

    @staticmethod
    def plot_regression_results(X: np.ndarray, y: np.ndarray) -> None:
        pass

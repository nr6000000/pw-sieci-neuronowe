import numpy as np


class Loss:
    @staticmethod
    def mae(y_target: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_target - y_pred))

    @staticmethod
    def mse(y_target: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_target - y_pred) ** 2)

    @staticmethod
    def bce(y_target: np.ndarray, y_pred: np.ndarray) -> float:
        pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(y_target * np.log(pred_clipped) + (1 - y_target) * np.log(1 - pred_clipped))

    class Gradient:
        @staticmethod
        def mae(y_target: np.ndarray, y_pred: np.ndarray) -> float:
            return np.sign(y_pred - y_target) / y_pred.shape[1]

        @staticmethod
        def mse(y_target: np.ndarray, y_pred: np.ndarray) -> float:
            return 2 * (y_pred - y_target) / y_pred.shape[1]

        @staticmethod
        def bce(y_target: np.ndarray, y_pred: np.ndarray) -> float:
            pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
            return (-y_target / pred_clipped + (1 - y_target) / (1 - pred_clipped)) / pred_clipped.shape[1]


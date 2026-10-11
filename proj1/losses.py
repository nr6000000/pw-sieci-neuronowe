import numpy as np


class Loss:
    @staticmethod
    def mae(y_batch: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_batch - y_pred))

    @staticmethod
    def mse(y_batch: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_batch - y_pred) ** 2)

    @staticmethod
    def bce(y_batch: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.mean(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))

    class Gradient:
        @staticmethod
        def mae(y_batch: np.ndarray, y_pred: np.ndarray) -> float:
            return np.sign(y_batch - y_pred)

        @staticmethod
        def mse(y_batch: np.ndarray, y_pred: np.ndarray) -> float:
            return 2 * (y_batch - y_pred) / y_pred.shape[1]

        @staticmethod
        def bce(y_batch: np.ndarray, y_pred: np.ndarray) -> float:
            pass


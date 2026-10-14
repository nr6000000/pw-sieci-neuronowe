import numpy as np


class Activation:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    class Gradient:
        @staticmethod
        def relu(x: np.ndarray) -> np.ndarray:
            return np.where(x <= 0, 0, 1) 

        @staticmethod
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))

        @staticmethod
        def tanh(x: np.ndarray) -> np.ndarray:
            return 1 - np.tanh(x) ** 2

        @staticmethod
        def linear(x: np.ndarray) -> np.ndarray:
            return np.ones(shape=x.shape)
        
        @staticmethod
        def softmax(x: np.ndarray) -> np.ndarray:
            raise NotImplementedError


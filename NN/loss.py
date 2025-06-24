import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):

    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass


class MSE(Loss):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.mean((y_pred - y_true) ** 2) / 2

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]


class BinaryCrossEntropy(Loss):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Prevent div by 0
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])


class SparseCategoricalCrossEntropy(Loss):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # prevent log(0)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]

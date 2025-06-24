import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):

    @abstractmethod
    def __call__(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, z: np.ndarray) -> np.ndarray:
        pass


class Linear(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


class ReLU(Activation):
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)


class Sigmoid(Activation):
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        sigmoid = self(z)
        return sigmoid * (1 - sigmoid)


class Softmax(Activation):
    def __call__(self, z: np.ndarray) -> np.ndarray:
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        z = z.reshape(-1, 1)
        return np.diagflat(z) - np.dot(z, z.T)


class Tanh:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        tanh = self(z)
        return 1 - tanh ** 2
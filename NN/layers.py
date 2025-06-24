import numpy as np
from abc import ABC, abstractmethod
from NN.activations import Activation, Softmax

class Layer(ABC):

    @abstractmethod
    def __init__(self, in_dim: int, out_dim: int):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray, lr: float) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self, in_dim: int, out_dim: int, activation: Activation):
        self.weights = np.random.random((in_dim, out_dim)) * np.sqrt(1 / in_dim)
        self.bias = np.zeros((1, out_dim))
        self.activation = activation
        self.x = None
        self.z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = self.x @ self.weights + self.bias
        return self.activation(self.z)

    def backward(self, dA: np.ndarray, lr: np.ndarray) -> np.ndarray:
        if isinstance(self.activation, Softmax):
            dz = dA
        else:
            dz = dA * self.activation.derivative(self.z)
        dw = self.x.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)

        # prevent exploding gradients
        np.clip(dw, -1, 1, out=dw)
        np.clip(db, -1, 1, out=db)

        self.weights -= lr * dw
        self.bias -= lr * db
        return dz @ self.weights.T

    @property
    def params_(self):
        return (self.weights, self.bias)
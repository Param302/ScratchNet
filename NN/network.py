import pickle
import numpy as np
from NN.layers import Layer
from NN.loss import Loss

class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss: Loss):
        self.layers = layers
        self.loss_fn = loss

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, lr: float):
        dA = self.loss_fn.derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, lr)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_X: np.ndarray = None,
        val_y: np.ndarray = None,
        n_iters: int = 100,
        lr: float = 0.01,
        verbose: bool = True
    ):
        self.history = []
        epoch_10pct = 1 if n_iters < 10 else int(n_iters * 0.1)

        for epoch in range(1, n_iters + 1):
            y_pred = self.forward(X)
            train_loss = self.loss_fn(y_pred, y)
            self.backward(y_pred, y, lr)

            if val_X is not None and val_y is not None:
                y_val_pred = self.forward(val_X)
                val_loss = self.loss_fn(y_val_pred, val_y)

            self.history.append(train_loss if val_X is None else (train_loss, val_loss))

            if verbose and epoch % epoch_10pct == 0:
                l = len(str(n_iters))
                print(
                    f"Epoch {epoch:{len(str(n_iters))}d} | Train Loss: {train_loss:.4f}" +
                    ("" if val_X is None else f" | Val Loss: {val_loss:.4f}")
                     )
        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @property
    def params_(self) -> np.ndarray:
        return [layer.params_ for layer in self.layers]

    def summary(self):
        print("Model Summary:")
        print("-" * 60)
        total_params = 0

        for i, layer in enumerate(self.layers, start=1):
            name = layer.__class__.__name__
            w_shape = layer.weights.shape
            b_shape = layer.bias.shape
            num_params = np.prod(w_shape) + np.prod(b_shape)
            total_params += num_params
            print(f"{name} layer {i}: Params: {num_params}"
                  f" ({w_shape[0]} x {w_shape[1]} + {b_shape[1]})")

        print("-" * 60)
        print(f"Total Layers: {i}")
        print(f"Total Trainable Parameters: {total_params}")
        print("-" * 60)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "NeuralNetwork":
        with open(filepath, "rb") as f:
            return pickle.load(f)

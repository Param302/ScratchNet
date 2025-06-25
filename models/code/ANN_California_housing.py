import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from NN.layers import Dense
from NN.network import NeuralNetwork
from NN.activations import ReLU, Linear
from NN.loss import MSE


def get_regression_data():
    data = fetch_california_housing()
    X = data.data
    y = data.target.reshape(-1, 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=37)
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_valid = std_scaler.transform(X_valid)
    return X_train, X_valid, y_train, y_valid

reg_X_train, reg_X_valid, reg_y_train, reg_y_valid = get_regression_data()

NN_reg = NeuralNetwork([
    Dense(reg_X_train.shape[1], 10, ReLU()),
    Dense(10, 16, ReLU()),
    Dense(16, 32, ReLU()),
    Dense(32, 16, ReLU()),
    Dense(16, 1, Linear())
],
    loss=MSE())

NN_reg.summary()

history = NN_reg.fit(
    reg_X_train,
    reg_y_train,
    reg_X_valid,
    reg_y_valid,
    n_iters=100
)

reg_y_preds = NN_reg.predict(reg_X_valid[10:15, :])
_ = reg_y_preds, reg_y_valid[10:15]
_, MSE()(*_)
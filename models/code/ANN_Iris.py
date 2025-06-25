import sys
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from NN.layers import Dense
from NN.network import NeuralNetwork
from NN.activations import ReLU, Softmax
from NN.loss import SparseCategoricalCrossEntropy


def get_classification_data():
    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=37)

    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_valid = std_scaler.transform(X_valid)

    ohe = OneHotEncoder(sparse_output=False)
    y_train = ohe.fit_transform(y_train)
    y_valid = ohe.transform(y_valid)
    return X_train, X_valid, y_train, y_valid

X_train, X_valid, y_train, y_valid = get_classification_data()

NN_clf = NeuralNetwork([
    Dense(X_train.shape[1], 8, ReLU()),
    Dense(8, 16, ReLU()),
    Dense(16, 64, ReLU()),
    Dense(64, 10, ReLU()),
    Dense(10, 3, Softmax())
],
    loss=SparseCategoricalCrossEntropy())

NN_clf.summary()

history = NN_clf.fit(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_iters=500
)

y_preds = NN_clf.predict(X_valid)
y_preds_lbl = np.argmax(y_preds, axis=1)
y_valid_lbl = np.argmax(y_valid, axis=1)
_ = y_preds_lbl, y_valid_lbl
_, SparseCategoricalCrossEntropy()(*_)

print(classification_report(y_preds_lbl, y_valid_lbl))
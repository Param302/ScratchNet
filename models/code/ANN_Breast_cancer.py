import sys
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from NN.layers import Dense
from NN.network import NeuralNetwork
from NN.activations import ReLU, Sigmoid
from NN.loss import BinaryCrossEntropy


def get_binary_clf_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=37, shuffle=True)

    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_valid = std_scaler.transform(X_valid)

    return X_train, X_valid, y_train, y_valid

X_train, X_valid, y_train, y_valid = get_binary_clf_data()

NN_clf2 = NeuralNetwork([
    Dense(X_train.shape[1], 30, ReLU()),
    Dense(30, 60, ReLU()),
    Dense(60, 15, ReLU()),
    Dense(15, 1, Sigmoid())
],
    loss=BinaryCrossEntropy())

print(NN_clf2.summary())

history = NN_clf2.fit(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_iters=100
)

y_preds = NN_clf2.predict(X_valid)
y_preds = (y_preds >= 0.5).astype(int)
BinaryCrossEntropy()(y_preds, y_valid)

print(classification_report(y_preds, y_valid))

# ANN Scratch Webapp

Made by Parampreet Singh
Email: connectwithparam.30@gmail.com



Califoria House Prediction
model: models/ANN_California_house.bin
Architecture:
NN_reg = NeuralNetwork([
    Dense(reg_X_train.shape[1], 10, ReLU()),
    Dense(10, 16, ReLU()),
    Dense(16, 32, ReLU()),
    Dense(32, 16, ReLU()),
    Dense(16, 1, Linear())
],
    loss=MSE())

NN_reg.summary()

Model Summary:
------------------------------------------------------------
Dense layer 1: Params: 90 (8 x 10 + 10)
Dense layer 2: Params: 176 (10 x 16 + 16)
Dense layer 3: Params: 544 (16 x 32 + 32)
Dense layer 4: Params: 528 (32 x 16 + 16)
Dense layer 5: Params: 17 (16 x 1 + 1)
------------------------------------------------------------
Total Layers: 5
Total Trainable Parameters: 1355
------------------------------------------------------------




Iris Flower Classification
model: models/ANN_Iris.bin

Architecture:
NN_clf = NeuralNetwork([
    Dense(X_train.shape[1], 8, ReLU()),
    Dense(8, 16, ReLU()),
    Dense(16, 64, ReLU()),
    Dense(64, 10, ReLU()),
    Dense(10, 3, Softmax())
],
    loss=SparseCategoricalCrossEntropy())

NN_clf.summary()

Model Summary:
------------------------------------------------------------
Dense layer 1: Params: 40 (4 x 8 + 8)
Dense layer 2: Params: 144 (8 x 16 + 16)
Dense layer 3: Params: 1088 (16 x 64 + 64)
Dense layer 4: Params: 650 (64 x 10 + 10)
Dense layer 5: Params: 33 (10 x 3 + 3)
------------------------------------------------------------
Total Layers: 5
Total Trainable Parameters: 1955
------------------------------------------------------------

Classification report: (on 20% data, random_state=37)
             precision    recall  f1-score   support

           0       1.00      1.00      1.00         9
           1       0.88      1.00      0.93         7
           2       1.00      0.93      0.96        14

    accuracy                           0.97        30
   macro avg       0.96      0.98      0.97        30
weighted avg       0.97      0.97      0.97        30


Breast Cancer Detection
model: models/ANN_Breast_cancer.bin

Architecture:


NN_clf2 = NeuralNetwork([
    Dense(X_train.shape[1], 30, ReLU()),
    Dense(30, 60, ReLU()),
    Dense(60, 15, ReLU()),
    Dense(15, 1, Sigmoid())
],
    loss=BinaryCrossEntropy())

NN_clf2.summary()

Model Summary:
------------------------------------------------------------
Dense layer 1: Params: 930 (30 x 30 + 30)
Dense layer 2: Params: 1860 (30 x 60 + 60)
Dense layer 3: Params: 915 (60 x 15 + 15)
Dense layer 4: Params: 16 (15 x 1 + 1)
------------------------------------------------------------
Total Layers: 4
Total Trainable Parameters: 3721
------------------------------------------------------------

              precision    recall  f1-score   support

           0       0.88      0.81      0.84        43
           1       0.89      0.93      0.91        71

    accuracy                           0.89       114
   macro avg       0.88      0.87      0.88       114
weighted avg       0.89      0.89      0.89       114
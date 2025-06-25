# [ScratchNet 🔗](https://scratchnet.streamlit.app)

## Overview

ScratchNet is a demonstration of building and training **Artificial Neural Networks** (ANNs) _from scratch in Python using only NumPy_ - no deep learning frameworks. The models are trained on classic datasets and this app showcases their performance and predictions interactively.

<video src="/assets/ScratchNet.mp4" controls width="100%" height="auto" poster="/assets/ScratchNet_thumbnail.png">
    Your browser does not support the video tag.
</video>

---

## Datasets

### Breast Cancer (Binary Classification)
Classifies tumors as benign or malignant using 30 features from digitized images of fine needle aspirate (FNA) of breast mass.  
**Target labels:** Benign (0), Malignant (1).  
**Dataset:** scikit-learn's Breast Cancer Wisconsin.

### California Housing (Regression)
Predicts the median house value in a California block group using 8 features such as median income, house age, average rooms, and location.  
**Target label:** MedHouseVal (Median House Value).  
**Dataset:** scikit-learn's California Housing.

### Iris Flower Species (Multiclass Classification)
Predicts the species of an iris flower using four features: sepal length, sepal width, petal length, and petal width.  
**Target labels:** Setosa, Versicolor, Virginica.  
**Dataset:** scikit-learn's Iris.

---

## Models

### Model Architecture

<details>
<summary><strong>Breast Cancer</strong></summary>

- 4 layers:  
  - Input: 30 features  
  - Hidden: 30, 60, 15 neurons  
  - Output: 1 (Sigmoid)
- Activations: ReLU, Sigmoid  
- Loss: Binary Cross Entropy  
- Accuracy: 89%  
[View Model Code](models/code/ANN_Breast_cancer.py)
</details>

<details>
<summary><strong>California Housing</strong></summary>

- 5 layers:  
  - Input: 8 features  
  - Hidden: 10, 16, 32, 16 neurons  
  - Output: 1 (Linear)
- Activations: ReLU, Linear  
- Loss: MSE  
[View Model Code](models/code/ANN_California_housing.py)
</details>

<details>
<summary><strong>Iris</strong></summary>

- 5 layers:  
  - Input: 4 features  
  - Hidden: 8, 16, 64, 10 neurons  
  - Output: 3 (Softmax)
- Activations: ReLU, Softmax  
- Loss: Sparse Categorical Cross Entropy  
- Accuracy: 97%  
[View Model Code](models/code/ANN_iris.py)
</details>

---

### Neural Network Module

The neural network, dense layers, loss, and activation functions are implemented from scratch in the [`NN/`](https://github.com/Param302/ScratchNet/tree/main/NN) directory:
- [`network.py`](https://github.com/Param302/ScratchNet/blob/main/NN/network.py): Neural network class, training loop, prediction, summary, save/load.
- [`layers.py`](https://github.com/Param302/ScratchNet/blob/main/NN/layers.py): Dense layer implementation.
- [`activations.py`](https://github.com/Param302/ScratchNet/blob/main/NN/activations.py): ReLU, Sigmoid, Softmax, Linear, Tanh.
- [`loss.py`](https://github.com/Param302/ScratchNet/blob/main/NN/loss.py): MSE, Binary Cross Entropy, Sparse Categorical Cross Entropy.

[View NN Module](NN/)

---

### Model Notebook

All model development, training, and analysis are documented in  
[ANN_from_Scratch.ipynb](ANN_from_Scratch.ipynb).

---

## Steps to Run

Requires:
- **Python** 3.10.16
- **Streamlit** 1.46.0

1. **Clone the repository**
   ```
   git clone https://github.com/Param302/ScratchNet.git
   cd ScratchNet
   ```

2. **Create a virtual environment**

   - **Windows**
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Linux/Mac**
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install requirements**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```
   streamlit run app/main.py
   ```

   **OR**

   **Run each model script directly** to train the model
   ```
   python models/code/ANN_Breast_cancer.py
   python models/code/ANN_California_housing.py
   python models/code/ANN_iris.py
   ```

---

## Contact

For any questions or contributions, feel free to reach out:  
[**Parampreet Singh**](https://parampreetsingh.me) 
Email: [connectwithparam.30@gmail.com](mailto:connectwithparam.30@gmail.com)

---


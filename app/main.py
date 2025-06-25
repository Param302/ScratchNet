import sys
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.tabs.iris import run as iris_run
from app.tabs.house_price import run as house_price_run
from app.tabs.breast_cancer import run as breast_cancer_run
from NN.layers import Dense
from NN.network import NeuralNetwork
from NN.activations import Linear, ReLU, Sigmoid, Softmax
from NN.loss import MSE, BinaryCrossEntropy, SparseCategoricalCrossEntropy




@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def get_data():
    house = fetch_california_housing(as_frame=True)
    iris = load_iris(as_frame=True)
    cancer = load_breast_cancer(as_frame=True)
    return house, iris, cancer


@st.cache_data
def preprocess_house_data(data):
    X = data['data']
    y = data['target']
    columns = X.columns.tolist()
    X_full = X.values if hasattr(X, 'values') else X
    y_full = y.values if hasattr(y, 'values') else y
    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=0.2, random_state=37)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_viz = pd.DataFrame(X_train, columns=columns)
    y_viz = pd.Series(y_train)
    return {
        'X_train': X_train,
        'y_train': y_train,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'X_viz': X_viz,
        'y_viz': y_viz,
        'columns': columns
    }


@st.cache_data
def preprocess_iris_data(data):
    X = data['data']
    y = data['target']
    columns = X.columns.tolist()
    X_full = X.values if hasattr(X, 'values') else X
    y_full = y.values if hasattr(y, 'values') else y
    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=0.2, random_state=37)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_viz = pd.DataFrame(X_train, columns=columns)
    y_viz = pd.Series(y_train)
    return {
        'X_train': X_train,
        'y_train': y_train,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'X_viz': X_viz,
        'y_viz': y_viz,
        'columns': columns
    }


@st.cache_data
def preprocess_cancer_data(data):
    X = data['data']
    y = data['target']
    columns = X.columns.tolist()
    X_full = X.values if hasattr(X, 'values') else X
    y_full = y.values if hasattr(y, 'values') else y
    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=0.2, random_state=37)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_viz = pd.DataFrame(X_train, columns=columns)
    y_viz = pd.Series(y_train)
    return {
        'X_train': X_train,
        'y_train': y_train,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'X_viz': X_viz,
        'y_viz': y_viz,
        'columns': columns
    }


@st.cache_data
def get_house_col_desc():
    return {
        'MedInc': 'Median income in block group',
        'HouseAge': 'Median house age in block group',
        'AveRooms': 'Average rooms per household',
        'AveOccup': 'Average household occupancy',
        'Latitude': 'Block group latitude value',
        'Longitude': 'Block group longitude value',
    }


@st.cache_data
def get_iris_col_desc():
    return {
        'sepal length (cm)': 'Sepal length in centimeters',
        'sepal width (cm)': 'Sepal width in centimeters',
        'petal length (cm)': 'Petal length in centimeters',
        'petal width (cm)': 'Petal width in centimeters',
    }


@st.cache_data
def get_cancer_col_desc():
    return {
        'mean radius': 'Mean radius of cell nuclei',
        'mean texture': 'Mean texture of cell nuclei',
        'mean perimeter': 'Mean perimeter of cell nuclei',
        'mean area': 'Mean area of cell nuclei',
        'mean smoothness': 'Mean smoothness of cell nuclei',
        'mean compactness': 'Mean compactness of cell nuclei',
        'mean concavity': 'Mean concavity of cell nuclei',
        'mean concave points': 'Mean concave points of cell nuclei',
        'mean symmetry': 'Mean symmetry of cell nuclei',
        'mean fractal dimension': 'Mean fractal dimension of cell nuclei',
        'radius error': 'Standard error of radius',
        'texture error': 'Standard error of texture',
        'perimeter error': 'Standard error of perimeter',
        'area error': 'Standard error of area',
        'smoothness error': 'Standard error of smoothness',
        'compactness error': 'Standard error of compactness',
        'concavity error': 'Standard error of concavity',
        'concave points error': 'Standard error of concave points',
        'symmetry error': 'Standard error of symmetry',
        'fractal dimension error': 'Standard error of fractal dimension',
        'worst radius': 'Worst (largest) radius',
        'worst texture': 'Worst (largest) texture',
        'worst perimeter': 'Worst (largest) perimeter',
        'worst area': 'Worst (largest) area',
        'worst smoothness': 'Worst (largest) smoothness',
        'worst compactness': 'Worst (largest) compactness',
        'worst concavity': 'Worst (largest) concavity',
        'worst concave points': 'Worst (largest) concave points',
        'worst symmetry': 'Worst (largest) symmetry',
        'worst fractal dimension': 'Worst (largest) fractal dimension',
    }


# Load models
model_dir = Path('models/bin')
house_model = load_model(model_dir / 'ANN_California_house.bin')
iris_model = load_model(model_dir / 'ANN_Iris.bin')
cancer_model = load_model(model_dir / 'ANN_Breast_cancer.bin')

# Load data
house_data, iris_data, cancer_data = get_data()

# Preprocess and cache everything
house_processed = preprocess_house_data(house_data)
iris_processed = preprocess_iris_data(iris_data)
cancer_processed = preprocess_cancer_data(cancer_data)
house_col_desc = get_house_col_desc()
iris_col_desc = get_iris_col_desc()
cancer_col_desc = get_cancer_col_desc()

st.set_page_config(page_title='ScratchNet', page_icon='🧠', layout='wide')
st.title('ScratchNet')
st.markdown("This app demonstrates custom-built **Neural Networks** implemented from _scratch in Python **using only NumPy**_.")

TABS = {
    '🏠 House Price Predictor': house_price_run,
    '🩺 Breast Cancer Classifier': breast_cancer_run,
    '🌸 Iris Flower Classifier': iris_run,
}
tab_keys = list(TABS.keys())
tab_objs = st.tabs(tab_keys)

for i, tab in enumerate(tab_objs):
    with tab:
        if i == 0:
            house_price_run(house_model, house_data,
                            house_processed, house_col_desc)
        elif i == 1:
            breast_cancer_run(cancer_model, cancer_data,
                              cancer_processed, cancer_col_desc)
        elif i == 2:
            iris_run(iris_model, iris_data, iris_processed, iris_col_desc)

st.markdown('<hr style="margin-top:2em;">', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:gray;">Made by Parampreet Singh</div>',
            unsafe_allow_html=True)

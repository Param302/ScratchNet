from NN.layers import Dense
from NN.network import NeuralNetwork
from NN.activations import Linear, ReLU, Sigmoid, Softmax
from NN.loss import MSE, BinaryCrossEntropy, SparseCategoricalCrossEntropy
from app.tabs.house_price import run as house_price_run
from app.tabs.breast_cancer import run as breast_cancer_run
from app.tabs.iris import run as iris_run
import sys
import pickle
import streamlit as st
from pathlib import Path
from sklearn.datasets import fetch_california_housing, load_iris, load_breast_cancer

sys.path.append(str(Path(__file__).resolve().parent.parent))


# Caching model loading


@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Caching data loading


@st.cache_data
def get_data():
    house = fetch_california_housing(as_frame=True)
    iris = load_iris(as_frame=True)
    cancer = load_breast_cancer(as_frame=True)
    return house, iris, cancer


# Load models
model_dir = Path('models')
house_model = load_model(model_dir / 'ANN_California_house.bin')
iris_model = load_model(model_dir / 'ANN_Iris.bin')
cancer_model = load_model(model_dir / 'ANN_Breast_cancer.bin')

# Load data
house_data, iris_data, cancer_data = get_data()

# Set page config
st.set_page_config(page_title='ScratchNet', page_icon='🧠', layout='wide')
st.title('ScratchNet')

# Tabs with icons
TABS = {
    '🏠 House Price Predictor': house_price_run,
    '🩺 Breast Cancer Classifier': breast_cancer_run,
    '🌸 Iris Flower Classifier': iris_run,
}
tab_keys = list(TABS.keys())
tab_objs = st.tabs(tab_keys)

for i, tab in enumerate(tab_objs):
    with tab:
        st.session_state['model'] = [house_model, cancer_model, iris_model][i]
        st.session_state['data'] = [house_data, cancer_data, iris_data][i]
        st.session_state['tab_name'] = list(TABS.values())[i] if hasattr(
            list(TABS.values())[i], '__str__') else tab_keys[i]
        list(TABS.values())[i]()
# Footer
st.markdown('<hr style="margin-top:2em;">', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:gray;">Made by Parampreet Singh</div>',
            unsafe_allow_html=True)


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import joblib
import pytest
import numpy as np
from sklearn.datasets import load_iris
from iris import train_model, evaluate_model, load_model

@pytest.fixture(scope="module")
def model():
    model = train_model()
    yield model
    joblib.dump(model, 'iris_model.pkl')

def test_train_model(model):
    assert model is not None
    assert hasattr(model, 'predict')

def test_evaluate_model(model):
    accuracy = evaluate_model(model)
    assert accuracy >= 0.9  # Espera-se uma precisão de pelo menos 90%

def test_load_model():
    model = load_model('iris_model.pkl')
    assert model is not None
    assert hasattr(model, 'predict')

def test_prediction_shape(model):
    iris = load_iris()
    sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample_data)
    assert prediction.shape == (1,)  # Deve retornar uma única previsão 

def test_prediction_value(model):
    sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample_data)
    assert prediction in [0, 1, 2]  # Classes alvo (0, 1, 2)


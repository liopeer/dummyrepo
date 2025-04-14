import pytest
import torch
from classification.neuralnetwork import SimpleMLP

@pytest.fixture
def model_params():
    return {
        'input_size': 10,
        'num_classes': 3
    }

@pytest.fixture
def model(model_params):
    return SimpleMLP(model_params['input_size'], model_params['num_classes'])

def test_model_initialization(model, model_params):
    assert isinstance(model, SimpleMLP)
    assert model.fc1.in_features == model_params['input_size']
    assert model.fc2.out_features == model_params['num_classes']

def test_forward_pass(model, model_params):
    # Test single sample
    x = torch.randn(1, model_params['input_size'])
    output = model(x)
    assert output.shape == (1, model_params['num_classes'])

    # Test batch of samples
    batch_size = 32
    x_batch = torch.randn(batch_size, model_params['input_size'])
    output_batch = model(x_batch)
    assert output_batch.shape == (batch_size, model_params['num_classes'])
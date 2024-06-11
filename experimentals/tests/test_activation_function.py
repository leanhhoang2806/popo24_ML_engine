import numpy as np

from experimentals.distributed_with_ray_from_sratch import calculate_activate, Sigmoid
import ray
import pytest


@pytest.fixture
def example_data():
    return np.array([[1.0, 2.0], [3.0, 4.0]])

def test_calculate_activate(example_data):
    result = ray.get(calculate_activate.remote(Sigmoid.activate, example_data))
    expected_result = Sigmoid.activate(example_data)
    assert np.allclose(result, expected_result), "The activation result is incorrect."

def test_sigmoid_derivative(example_data):
    result = Sigmoid.derivative(example_data)
    sigmoid_output = Sigmoid.activate(example_data)
    expected_derivative = sigmoid_output * (1 - sigmoid_output)
    assert np.allclose(result, expected_derivative), "The derivative result is incorrect."

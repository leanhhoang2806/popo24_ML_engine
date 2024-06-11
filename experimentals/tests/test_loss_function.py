import numpy as np
import ray
import pytest

from experimentals.distributed_with_ray_from_sratch import calculate_loss


@pytest.fixture
def example_data():
    prediction = np.array([[1.0, 2.0],
                           [3.0, 4.0]])

    target = np.array([[0.0, 1.0],
                       [2.0, 3.0]])

    return prediction, target

def test_calculate_loss(example_data):
    prediction, target = example_data
    loss = ray.get(calculate_loss.remote(prediction, target))
    assert np.isscalar(loss)  # Check if the loss is a scalar value

def test_calculate_loss_with_keep_shape(example_data):
    prediction, target = example_data
    loss = ray.get(calculate_loss.remote(prediction, target, keep_shape=True))
    assert loss.shape == prediction.shape  # Check if the shape of the loss matches the input arrays

if __name__ == "__main__":
    pytest.main()

import pytest
import tensorflow as tf
from experimentals.distributed_with_ray_from_sratch import series_matrix_multiplication
import ray
import numpy as np


@pytest.fixture
def example_data():
    array = np.array([[1.0, 2.0]])

    matrix1 = np.random.randn(2, 10)  # First matrix with 10 features
    matrix2 = np.random.randn(10, 5)  # Second matrix with 5 features
    matrix3 = np.random.randn(5, 2)   # Third matrix with 2 features, same as input array

    return array, [matrix1, matrix2, matrix3]

def test_series_matrix_multiplication(example_data):
    array, matrix_list = example_data
    result_ref = np.matmul(np.matmul(np.matmul(array, matrix_list[0]), matrix_list[1]), matrix_list[2])
    result_ray = ray.get(series_matrix_multiplication.remote(array, matrix_list))
    assert np.array_equal(result_ref, result_ray)
    assert result_ray.shape == array.shape
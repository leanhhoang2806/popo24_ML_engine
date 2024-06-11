import numpy as np
import ray
from typing import List
from abc import ABC, abstractmethod

ray.init()

@ray.remote(num_gpus=1)
def series_matrix_multiplication(array: np.ndarray, matrix_list: List[np.ndarray]) -> np.ndarray:
    # Check if GPU is available
    if ray.get_gpu_ids():
        print("Function is executed on GPU.")
    else:
        print("Function is executed on CPU.")

    # Check if the shapes of matrices in matrix_list are compatible
    input_shape = array.shape[1]
    for i, matrix in enumerate(matrix_list):
        if matrix.shape[0] != input_shape:
            raise ValueError(f"The shape of matrix {i+1} in matrix_list does not match the expected shape.")

        input_shape = matrix.shape[1]

    # Perform matrix multiplications
    result = array
    for matrix in matrix_list:
        result = np.matmul(result, matrix)

    # Convert the result back to the input array shape if necessary
    if result.shape != array.shape:
        result = np.matmul(result, np.ones((input_shape, array.shape[1])))

    return result


@ray.remote(num_gpus=1)
def calculate_loss(prediction: np.ndarray, target: np.ndarray, keep_shape: bool = False) -> np.ndarray:
    """
    Calculate loss between prediction and target arrays.

    Args:
        prediction (np.ndarray): The prediction array.
        target (np.ndarray): The target array.
        keep_shape (bool, optional): Whether to keep the shape of the output loss 
            the same as the input arrays. Defaults to False.

    Returns:
        np.ndarray: The loss array.
    """
    # Check if the shapes of the arrays are compatible
    if prediction.shape != target.shape:
        raise ValueError("The shapes of the prediction and target arrays do not match.")

    # Calculate loss (for example, using mean squared error)
    loss = np.mean(np.square(prediction - target))

    # Optionally, ensure that the shape of the output loss is the same as the input arrays
    if keep_shape:
        loss = np.ones_like(prediction) * loss

    return loss


class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

class Sigmoid(ActivationFunction):
    @classmethod
    def activate(cls, x: np.ndarray) -> np.ndarray:
        """Apply the sigmoid activation function.

        The sigmoid function is defined as 1 / (1 + exp(-x)).

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
    
    @classmethod
    def derivative(cls, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the sigmoid function.

        The derivative of the sigmoid function is sigmoid(x) * (1 - sigmoid(x)).

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The derivative of the sigmoid function.
        """
        sigmoid_x = cls.activate(x)
        return sigmoid_x * (1 - sigmoid_x)


@ray.remote
def calculate_activate(activation_func, x: np.ndarray) -> np.ndarray:
    """Calculate the activation of an input array using the specified activation function.

    Args:
        activation_func (Callable[[np.ndarray], np.ndarray]): The activation function to apply.
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The output array after applying the activation function.
    """
    # Check if GPU is available
    if ray.get_gpu_ids():
        print("Function is executed on GPU.")
    else:
        print("Function is executed on CPU.")

    return activation_func(x)
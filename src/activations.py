"""Activation function placeholders for the neural network."""

import numpy as np
from typing import Callable


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation placeholder.

    TODO: implement the sigmoid function for forward propagation.
    """
    # TODO: return 1 / (1 + np.exp(-x))
    return np.zeros_like(x)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation placeholder.

    TODO: implement the tanh function for forward propagation.
    """
    return np.zeros_like(x)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation placeholder.

    TODO: implement the ReLU function for forward propagation.
    """
    return np.zeros_like(x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation placeholder.

    TODO: implement the softmax function for multiclass output.
    """
    return np.zeros_like(x)


def get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return an activation function by name."""
    # TODO: add support for activation mapping.
    activations = {
        "sigmoid": sigmoid,
        "tanh": tanh,
        "relu": relu,
        "softmax": softmax,
    }
    return activations.get(name.lower(), relu)

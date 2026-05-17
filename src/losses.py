"""Loss function placeholders for the neural network."""

import numpy as np
from typing import Tuple


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross entropy loss placeholder.

    TODO: implement categorical cross entropy for model evaluation.
    """
    return 0.0


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error placeholder.

    TODO: implement MSE for regression or auxiliary metrics.
    """
    return 0.0


def loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Placeholder for loss gradient calculation.

    TODO: implement derivative for backpropagation.
    """
    return np.zeros_like(y_pred)


def get_loss(name: str):
    """Return a loss function or loss handler by name."""
    # TODO: map name to loss functions.
    losses = {
        "cross_entropy": cross_entropy_loss,
        "mse": mean_squared_error,
    }
    return losses.get(name.lower(), cross_entropy_loss)

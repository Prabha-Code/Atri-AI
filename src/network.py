"""Neural network architecture placeholders."""

import numpy as np
from typing import Any, Dict, List, Optional

from src.activations import get_activation
from src.losses import get_loss
from src.utils import initialize_weights


class NeuralNetwork:
    """Configurable neural network container for NumPy training."""

    def __init__(self, architecture: List[int], config: Dict[str, Any]) -> None:
        self.architecture = architecture
        self.config = config
        self.layers: List[Dict[str, np.ndarray]] = []
        self.loss_function = get_loss(config.get("loss", "cross_entropy"))
        self.activation_name = config.get("activation", "relu")
        self.activation = get_activation(self.activation_name)
        self.initialize_network()

    def initialize_network(self) -> None:
        """Initialize model parameters using weight initialization strategies."""
        # TODO: use Xavier or random initialization based on config.
        self.layers = initialize_weights(self.architecture, self.config)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation placeholder.

        TODO: implement forward propagation through each layer.
        """
        raise NotImplementedError

    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Backpropagation placeholder.

        TODO: compute gradients and update layer state for training.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for given input data."""
        return self.forward(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on a dataset."""
        # TODO: compute loss and metrics for evaluation.
        raise NotImplementedError

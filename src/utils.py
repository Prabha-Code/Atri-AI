"""Utility functions for model configuration and initialization."""

import numpy as np
from typing import Any, Dict, List


def initialize_weights(architecture: List[int], config: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    """Initialize network weights and biases.

    TODO: implement Xavier and random initialization strategies.
    """
    print("[utils] Initializing network weights placeholder")
    layers = []
    for _ in range(len(architecture) - 1):
        layers.append({"weights": np.zeros((0, 0)), "biases": np.zeros((0,))})
    return layers


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    # TODO: control reproducibility for NumPy operations.
    np.random.seed(seed)
    print(f"[utils] Seed set to {seed}")


def get_network_architecture(config: Dict[str, Any]) -> List[int]:
    """Construct the network architecture based on configuration."""
    # TODO: parse hidden layers and output dimensions.
    return config.get("architecture", [784, 128, 10])

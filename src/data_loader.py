"""Data loading utilities for Fashion-MNIST."""

from typing import Any, Dict, Tuple

import numpy as np


class DataLoader:
    """Data loader placeholder for dataset ingestion."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.train_data: Tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0))
        self.test_data: Tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0))

    def load_data(self) -> None:
        """Load Fashion-MNIST data from disk or remote source."""
        # TODO: implement dataset download, normalization, and encoding.
        print("[data_loader] Loading data placeholder")
        self.train_data = (np.empty((0, 784)), np.empty((0,)))
        self.test_data = (np.empty((0, 784)), np.empty((0,)))

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return a single training batch."""
        # TODO: return the next batch for training.
        print("[data_loader] Returning empty batch placeholder")
        return np.empty((0, 784)), np.empty((0,))

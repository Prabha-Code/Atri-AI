"""Training workflow placeholder for the neural network."""

from typing import Any, Dict

from src.data_loader import DataLoader
from src.network import NeuralNetwork
from src.optimizers import get_optimizer
from src.metrics import compute_accuracy, plot_confusion_matrix


class Trainer:
    """Training engine placeholder for a NumPy neural network."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.data_loader = DataLoader(config)
        self.model = NeuralNetwork(config.get("architecture", [784, 128, 10]), config)
        self.optimizer = get_optimizer(config.get("optimizer", "sgd"), config)

    def prepare_data(self) -> None:
        """Load and preprocess the Fashion-MNIST dataset."""
        # TODO: implement data loading and batch creation.
        self.data_loader.load_data()

    def train(self) -> None:
        """Run the full training loop."""
        # TODO: implement epoch loop, forward pass, backward pass, and optimizer updates.
        print("[trainer] Starting training placeholder")
        print("[trainer] Training is currently a scaffold and does not perform real updates.")

    def evaluate(self) -> None:
        """Evaluate the model and log metrics."""
        # TODO: implement evaluation using validation/test split.
        print("[trainer] Evaluation placeholder")
        print("[trainer] No actual model metrics are computed in this scaffold.")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics to an external service such as wandb."""
        # TODO: add wandb logging hooks here.
        print(f"[trainer] Logging metrics: {metrics}")

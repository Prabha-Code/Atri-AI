"""Training runner for the ATRI Fashion-MNIST neural network."""

from typing import Any, Dict

from src.trainer import Trainer
from src.wandb_sweep import init_wandb


def run_training(config: Dict[str, Any]) -> None:
    """Run a training session using a provided configuration."""
    init_wandb(config)
    trainer = Trainer(config)
    # TODO: implement dataset loading and trainer lifecycle calls.
    trainer.prepare_data()
    trainer.train()
    trainer.evaluate()

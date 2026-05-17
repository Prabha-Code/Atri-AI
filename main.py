"""Main entry point for the ATRI Fashion-MNIST neural network scaffold."""

from src.train import run_training


def main() -> None:
    """Run the main application workflow."""
    # TODO: Add CLI parsing and configuration handling.
    config = {
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "hidden_layers": [128, 64],
        "activation": "relu",
        "optimizer": "adam",
    }
    run_training(config)


if __name__ == "__main__":
    main()

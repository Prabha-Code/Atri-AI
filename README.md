# ATRI Assignment

A clean Python project scaffold for a neural network using NumPy to classify Fashion-MNIST.

## Project Structure

- `data/` - dataset downloads and local artifacts
- `notebooks/` - exploratory analysis notebooks
- `src/` - project source code
  - `data_loader.py`
  - `activations.py`
  - `losses.py`
  - `network.py`
  - `optimizers.py`
  - `trainer.py`
  - `metrics.py`
  - `utils.py`
  - `wandb_sweep.py`
- `results/` - output charts, checkpoints, and reports
- `report/` - assignment deliverables
- `main.py` - application entry point
- `train.py` - training runner
- `requirements.txt` - required Python dependencies
- `.gitignore` - ignore rules

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   ```bash
   source venv/bin/activate  # Linux / macOS
   venv\\Scripts\\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- `python main.py` - main application launcher
- `python train.py` - skeleton training runner
- `python src/wandb_sweep.py` - placeholder sweep launcher

## Notes

- The repository uses NumPy-only neural network components.
- All implementations are skeletons with TODO placeholders.
- Backpropagation and forward propagation should be implemented inside `src/network.py` and `src/trainer.py`.

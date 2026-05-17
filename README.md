## Project Description

This project demonstrates the implementation of a Feedforward Neural Network from scratch using NumPy for Fashion-MNIST image classification.

The neural network supports:

* Flexible hidden layers
* Multiple activation functions
* Multiple optimizers
* Hyperparameter tuning using WandB
* Visualization and evaluation metrics

---

# Features

* Forward Propagation
* Backpropagation
* SGD Optimizer
* Momentum Optimizer
* Nesterov Optimizer
* RMSProp Optimizer
* Adam Optimizer
* Nadam Optimizer
* Cross Entropy Loss
* Squared Error Loss
* WandB Hyperparameter Sweeps
* Confusion Matrix Visualization
* MNIST Transfer Learning Analysis

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* WandB
* TensorFlow (dataset loading only)

---

# Dataset

Fashion-MNIST dataset:

* 60,000 training images
* 10,000 test images
* 10 classes
* 28 × 28 grayscale images

Dataset loaded using:

python
from tensorflow.keras.datasets import fashion_mnist


---

# How to Run the Project

## Step 1

Clone the repository

bash
git clone https://github.com/your-username/fashion-mnist-neural-network.git


## Step 2

Move into the project folder

bash
cd fashion-mnist-neural-network


## Step 3

Install dependencies

bash
pip install -r requirements.txt


## Step 4

Run Jupyter Notebook

bash
jupyter notebook


---

# Training the Model

Open:

text
notebooks/task3_backpropagation_optimizers.ipynb


Run all cells to train the model.

---

# Hyperparameter Tuning

Open:

text
notebooks/task4_wandb_sweeps.ipynb


Run WandB sweep experiments.

---

# Evaluation

Open:

text
notebooks/task7_confusion_matrix.ipynb


This notebook generates:

* Test Accuracy
* Classification Report
* Confusion Matrix
* Visualization Plots

---

# Best Performing Configuration

| Hyperparameter    | Value |
| ----------------- | ----- |
| Optimizer         | Adam  |
| Activation        | ReLU  |
| Hidden Layers     | 4     |
| Hidden Layer Size | 128   |
| Learning Rate     | 0.001 |
| Batch Size        | 32    |
| Epochs            | 10    |

---

# Results

* Validation Accuracy achieved: ~90%+
* Strong convergence using Adam optimizer
* ReLU activation provided best performance
* Cross entropy loss outperformed squared error loss

---

# Conclusion

The project successfully demonstrates:

* implementation of neural networks from scratch,
* backpropagation using NumPy,
* optimization algorithms,
* hyperparameter tuning,
* and transfer learning insights.

The experiments show that careful hyperparameter tuning significantly improves classification performance on image datasets.

---

# Author

Prabha

Artificial Intelligence and Data Science

---

# License

This project is developed for educational and research purposes.

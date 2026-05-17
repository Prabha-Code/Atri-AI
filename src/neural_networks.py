import numpy as np

from activations import relu
from activations import relu_derivative
from activations import softmax


class FeedForwardNeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size
    ):

        self.layers = [input_size] + hidden_layers + [output_size]

        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):

            weight = np.random.randn(
                self.layers[i],
                self.layers[i + 1]
            ) * np.sqrt(1 / self.layers[i])

            bias = np.zeros((1, self.layers[i + 1]))

            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):

        activations = [X]
        z_values = []

        A = X

        for i in range(len(self.weights) - 1):

            Z = np.dot(A, self.weights[i]) + self.biases[i]

            z_values.append(Z)

            A = relu(Z)

            activations.append(A)

        Z = np.dot(A, self.weights[-1]) + self.biases[-1]

        z_values.append(Z)

        output = softmax(Z)

        activations.append(output)

        return activations, z_values

    def backward(
        self,
        X,
        y,
        activations,
        z_values
    ):

        m = X.shape[0]

        gradients_w = []
        gradients_b = []

        dZ = activations[-1] - y

        for i in reversed(range(len(self.weights))):

            dW = np.dot(activations[i].T, dZ) / m

            dB = np.sum(dZ, axis=0, keepdims=True) / m

            gradients_w.insert(0, dW)
            gradients_b.insert(0, dB)

            if i > 0:

                dA = np.dot(dZ, self.weights[i].T)

                dZ = dA * relu_derivative(z_values[i - 1])

        return gradients_w, gradients_b


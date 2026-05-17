from neural_network import FeedForwardNeuralNetwork
from losses import cross_entropy_loss
from utils import accuracy


def train_model(
    x_train,
    y_train,
    y_train_encoded
):

    model = FeedForwardNeuralNetwork(
        input_size=784,
        hidden_layers=[128, 64],
        output_size=10
    )

    epochs = 10
    learning_rate = 0.001

    for epoch in range(epochs):

        activations, z_values = model.forward(x_train)

        loss = cross_entropy_loss(
            y_train_encoded,
            activations[-1]
        )

        gradients_w, gradients_b = model.backward(
            x_train,
            y_train_encoded,
            activations,
            z_values
        )

        for i in range(len(model.weights)):

            model.weights[i] -= learning_rate * gradients_w[i]
            model.biases[i] -= learning_rate * gradients_b[i]

        train_accuracy = accuracy(
            y_train,
            activations[-1]
        )

        print(
            f"Epoch {epoch+1} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {train_accuracy:.4f}"
        )


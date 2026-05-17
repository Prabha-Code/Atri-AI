import numpy as np


class Optimizers:

    def __init__(self, model):

        self.model = model

    def sgd(self, gradients_w, gradients_b, learning_rate):

        for i in range(len(self.model.weights)):

            self.model.weights[i] -= learning_rate * gradients_w[i]

            self.model.biases[i] -= learning_rate * gradients_b[i]


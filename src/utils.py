python
import numpy as np


def one_hot_encode(y, num_classes=10):

    one_hot = np.zeros((y.size, num_classes))

    one_hot[np.arange(y.size), y] = 1

    return one_hot


def accuracy(y_true, y_pred):

    predictions = np.argmax(y_pred, axis=1)

    return np.mean(predictions == y_true)

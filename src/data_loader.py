from tensorflow.keras.datasets import fashion_mnist



def load_fashion_mnist():

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 784) / 255.0

    x_test = x_test.reshape(x_test.shape[0], 784) / 255.0

    return x_train, y_train, x_test, y_test

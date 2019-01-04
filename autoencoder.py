"""
This programm tries with the help of an autoencoder to encode images from the
MNIST dataset to a lower dimension and decode them as good as possible.
It will be interesting to see what the lowest encoding dimesion can be without
loosing a lot of accuracy.

Training data:
    X: (60000, 784)
    y: (60000, 1)

Test data:
    X: (10000, 784)
    y: (10000, 1)
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense

import mnist

X_train, y_train, X_test, y_test = mnist.load()

# this works relatively well.
EPOCHS = 15
BATCH_SIZE = 64
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'kullback_leibler_divergence'
ACTIVATION_FUNCTION = 'relu' # the last activation needs to be softmax.

# how many nodes for each layer.
# decoder
INPUT_LAYER = 784
ENCODER_LAYER_1 = 256
ENCODER_LAYER_2 = 128
ENCODER_LAYER_3 = 64
# encoder
DECODER_LAYER_1 = 128
DECODER_LAYER_2 = 256
DECODER_LAYER_3 = 784

def normalize(X):
    """Normalizes X.

    Args:
        X (np.array): The image as an array. Shape=(784,).

    Returns:
        X (np.array): The image as an array normalized. Shape=(784,).
        normalizer (sklearn normalizer object): Normalizer object.
    """
    normalizer = Normalizer()
    X = normalizer.fit_transform(X)

    return X, normalizer

def show_digit(X, digit):
    """Visualizes one array as an image from the MNIST dataset.

    Args:
        X (np.array): The image as an array. Shape=(784,).
        digit (int): The digit shown in the image.
    """
    pixels = X.reshape((28, 28))
    plt.title(str(digit))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def show_digit_before_after(X, digit, autoencoder, normalizer):
    """Visualizes one digit before encoding and another one after decoding.

    Args:
        X (np.array): The image as an array. Shape=(784,).
        digit (int): The digit shown in the image.
        autoencoder (keras.models): The trained autoencoder.
        normalizer (sklearn.preprocessing.Normalizer): Object used for
            normalization.
    """
    # image before.
    show_digit(X, digit)

    # image after.
    X = X.reshape(1,784)

    # normalize.
    X = normalizer.transform(X)
    image_after = autoencoder.predict(X)
    show_digit(image_after, digit)


def train(X):
    """Trains the autoencoder.

    Args:
        X (np.array): The images. Shape(60000, 784)
    """
    autoencoder = Sequential()
    autoencoder.add(Dense(units=ENCODER_LAYER_1, activation=ACTIVATION_FUNCTION, input_dim=INPUT_LAYER))
    autoencoder.add(Dense(units=ENCODER_LAYER_2, activation=ACTIVATION_FUNCTION))
    autoencoder.add(Dense(units=ENCODER_LAYER_3, activation=ACTIVATION_FUNCTION))
    autoencoder.add(Dense(units=DECODER_LAYER_1, activation=ACTIVATION_FUNCTION))
    autoencoder.add(Dense(units=DECODER_LAYER_2, activation=ACTIVATION_FUNCTION))
    autoencoder.add(Dense(units=DECODER_LAYER_3, activation='softmax'))

    autoencoder.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])
    autoencoder.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE)

    return autoencoder

if __name__ == '__main__':
    # normalize.
    X_train, normalizer = normalize(X_train)
    autoencoder = train(X_train)

    digit_1 = X_train[1]
    label_1 = y_train[1]

    digit_2 = X_train[6]
    label_2 = y_train[6]

    digit_3 = X_train[16]
    label_3 = y_train[16]

    show_digit_before_after(digit_1, label_1, autoencoder, normalizer)
    show_digit_before_after(digit_2, label_2, autoencoder, normalizer)
    show_digit_before_after(digit_3, label_3, autoencoder, normalizer)

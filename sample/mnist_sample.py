import numpy as np
from PIL import Image
from keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from NEBULA import Injector, loadFatModel

if __name__ == '__main__':

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data: Normalize and reshape
    x_test = x_test.astype('float32') / 255  # Normalize pixel values between 0 and 1
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # Reshape for CNN input

    # Convert labels to one-hot encoding
    y_test = to_categorical(y_test, 10)

    # Load the pre-trained model from .h5 file either with or without FAT
    model = loadFatModel('sampledata/fat_mnist_model.h5', compile=False)



    model.summary()

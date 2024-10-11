import tensorflow as tf
from keras.datasets import mnist
from keras.src.models.cloning import clone_model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from PIL import Image
from skimage import transform

import numpy as np

from NEBULA.core.injector import Injector

if __name__ == '__main__':

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data: Normalize and reshape
    x_test = x_test.astype('float32') / 255  # Normalize pixel values between 0 and 1
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # Reshape for CNN input

    # Convert labels to one-hot encoding
    y_test = to_categorical(y_test, 10)

    # Load the pre-trained model from .h5 file
    model = load_model('sampledata/mnist_model.h5', compile=False)
    weightsBefore = model.get_weights()

    injector = Injector(model, probability=0.0001)
    model = injector.injectError()
    weightsAfter = model.get_weights()

    # Preprocess the image data
    # Step 1: Load and resize image to 28x28 pixels
    img = Image.open('sampledata/six.png').resize((28, 28))

    # Step 2: Convert to grayscale
    img = img.convert('L')  # 'L' mode means grayscale

    # Step 3: Convert to NumPy array and normalize pixel values (0-1 range)
    img_array = np.array(img) / 255.0

    # Step 4: Reshape the array to match MNIST model input (28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    preds = model.predict(img_array)
    print(preds)

    max = np.max(preds)
    output = np.where(preds == max)
    print(f"Prediction: {output[1]}, certainty: {max}")

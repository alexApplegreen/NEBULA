import csv
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

from NEBULA.core import Injector, ErrorTypes

SAMPLESIZE = 1e6  # million tries

if __name__ == "__main__":

    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data: Normalize and reshape
    x_test = x_test.astype('float32') / 255.0  # Normalize pixel values between 0 and 1
    x_test = x_test.reshape(-1, 28, 28, 1)  # Reshape for CNN input

    model = load_model("../sample/sampledata/mnist_model.h5", compile=False);
    # Recompile the model with valid metrics
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    injector = Injector(model.layers)

    with open("./results_acc_mnist.csv", "w+") as file:
        csvwriter = csv.writer(file)
        counter = 1
        for ber in np.linspace(0.0, 3e-06, SAMPLESIZE):
            injector.probability = ber
            injector.injectError(model, ErrorTypes.NORMAL)
            score = model.evaluate(x_test, y_test, verbose=0)
            injector.undo(model)
            print(f"cycle{counter}/{SAMPLESIZE} BER: {ber}% accuracy: {score[1]}")
            csvwriter.writerow([ber, score[1]])
            counter += 1

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from NEBULA import TrainingInjector

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data (normalize and reshape)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add a channel dimension (for convolutional layers)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Create a neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for MNIST
])

# inject errors during training
ti = TrainingInjector(probability=0.005, clipping=(-1.0, 1.0))
model = ti.attach(model, 3)

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Save the trained model to an h5 file
model.save('sampledata/fat_mnist_model.h5')

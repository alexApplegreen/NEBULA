import tensorflow as tf
from keras.datasets import mnist
from keras.src.models.cloning import clone_model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data: Normalize and reshape
x_test = x_test.astype('float32') / 255  # Normalize pixel values between 0 and 1
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # Reshape for CNN input

# Convert labels to one-hot encoding
y_test = to_categorical(y_test, 10)

# Load the pre-trained model from .h5 file
model = load_model('MNIST_UTIL/mnist_model.h5')
# model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy')

# Evaluate the model on the MNIST test data
score = model.evaluate(x_test, y_test, verbose=0)
print(f"evaluate original model: {score}")

modelCopy = clone_model(model)
modelCopy.set_weights(model.get_weights())

score = model.evaluate(x_test, y_test, verbose=0)
print(f"evaluate original model: {score}")
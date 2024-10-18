from logging import Logger
from keras.src import Layer

import numpy as np

import tensorflow as tf

from NEBULA.utils.logging import getLogger


class NoiseLayer(Layer):

    _logger: Logger
    _probability: float
    _inputShape: np.shape

    def __init__(self, probability=0.01, **kwargs):
        super().__init__(**kwargs)
        self._logger = getLogger(__name__)
        self._probability = probability

    def build(self, input_shape):
        self._inputShape = input_shape

        self.noise = self.add_weight(
            shape=input_shape[1:],  # match the shape of the previous layer's output
            initializer='zeros',
            trainable=False,
            name='noise_weight'
        )
        super(NoiseLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            # Inject noise only during training
            noise = self._probability * tf.random.normal(shape=tf.shape(inputs))
            return inputs + noise
        return inputs  # During inference, no noise is added

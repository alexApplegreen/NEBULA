#!/usr/bin/env python3

"""
NoiseLayer.py
    Class to mimic noise during training
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from logging import Logger
from keras.src import Layer

import numpy as np

import tensorflow as tf

from NEBULA.utils.logging import getLogger


class NoiseLayer(Layer):
    """subclass of keras layer simulating noise during training
    This class is a subclass of keras.layers.layer
    and can be used as such.
    It can be added to any model and will take the shape of the previous layer
    During training, the weights will add noise to the network during backpropagation
    """

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

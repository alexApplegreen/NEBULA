#!/usr/bin/env python3

"""
noiseLayer.py
    Class to mimic noise during training
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from logging import Logger
from keras.src import Layer

import tensorflow as tf

from NEBULA.utils.logging import getLogger


class NoiseLayer(Layer):
    # TODO must be tested
    """subclass of keras layer simulating noise during training
    This class is a subclass of keras.layers.layer
    and can be used as such.
    It can be added to any model and will take the shape of the previous layer
    During training, the weights will add noise to the network during backpropagation
    """

    _logger: Logger
    _errorProbability: float

    def __init__(self, probability: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self._logger = getLogger(__name__)
        self._errorProbability = probability

    def call(self, inputs, training=None):
        if training:
            error_mask = tf.random.uniform(tf.shape(inputs)) < self._errorProbability
            errors = tf.cast(error_mask, dtype=inputs.dtype)
            corrupted_inputs = inputs + errors
            return corrupted_inputs

        return inputs  # During inference, no noise is added

    @property
    def probability(self) -> float:
        return self._errorProbability

    @probability.setter
    def probability(self, probability: float) -> None:
        if probability < .0:
            raise ValueError("Probablility cannot be negative")
        self._errorProbability = probability

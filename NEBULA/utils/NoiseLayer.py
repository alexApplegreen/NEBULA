#!/usr/bin/env python3

"""
noiseLayer.py
    Class to mimic noise during training
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from logging import Logger

import numpy as np
from keras.src import Layer
from keras.src.layers import Flatten

import tensorflow as tf

from NEBULA.utils.logging import getLogger
from NEBULA.utils.commons import flipFloat, flipTensorBits


class NoiseLayer(Layer):
    """subclass of keras layer simulating noise during training
    This class is a subclass of keras.layers.layer
    and can be used as such.
    It can be added to any model and will take the shape of the previous layer
    During training, the weights will add noise to the network during forwardpropagation
    """

    _logger: Logger
    _errorProbability: float

    def __init__(self, probability: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False
        self._logger = getLogger(__name__)
        self._errorProbability = probability

    def call(self, inputs, training=None):
        """Injects noise into model during training
        This method is called by keras during traning.
        While feeding the data through the model (before evaluating the loss function)
        this will take the values from the preceeding layers and modfiy them with a given probability.
        This will pertubate the results of the model during training.
        """
        if training is True:
            self._logger.debug(f"injecting errors during training with BER of {self._errorProbability}")
            results = tf.map_fn(self._outerHelper, inputs)
            return results

        return inputs  # During inference, no noise is added

    def _outerHelper(self, x):
        return flipTensorBits(x, probability=self._errorProbability, dtype=np.float32)

    @property
    def probability(self) -> float:
        return self._errorProbability

    @probability.setter
    def probability(self, probability: float) -> None:
        if probability < .0:
            raise ValueError("Probablility cannot be negative")
        self._errorProbability = probability

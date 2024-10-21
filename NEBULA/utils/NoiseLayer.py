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

import tensorflow as tf

from NEBULA.utils.logging import getLogger
from NEBULA.utils.commons import flipFloat


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
        if training:
            self._logger.debug(f"injecting errors during training with BER of {self._errorProbability}")
            # TODO clashes with tensorflows graph stuff
            return [flipFloat(x, probability=self._errorProbability) for x in inputs]

        return inputs  # During inference, no noise is added

    @property
    def probability(self) -> float:
        return self._errorProbability

    @probability.setter
    def probability(self, probability: float) -> None:
        if probability < .0:
            raise ValueError("Probablility cannot be negative")
        self._errorProbability = probability

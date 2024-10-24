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
import tensorflow as tf
from keras.src import Layer

from NEBULA.utils.commons import flipTensorBits
from NEBULA.utils.logging import getLogger


class NoiseLayer(Layer):
    """subclass of keras layer simulating noise during training
    This class is a subclass of keras.layers.layer
    and can be used as such.
    It can be added to any model and will take the shape of the previous layer
    During training, the weights will add noise to the network during forwardpropagation
    """

    _logger: Logger
    _errorProbability: float
    _clipping: None | tuple = None

    def __init__(self, probability: float = 0.01, clipping: tuple | None = None, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False
        self._logger = getLogger(__name__)
        self._errorProbability = probability
        self._parseClipping(clipping)

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
            if self._clipping is not None:
                self._logger.debug(f"clipping enabled with min: {self._clipping[0]}, min: {self._clipping[1]}")
                results = tf.clip_by_value(results, self._clipping[0], self._clipping[1])
            return results

        return inputs  # During inference, no noise is added

    def _outerHelper(self, x):
        return flipTensorBits(x, probability=self._errorProbability, dtype=np.float32)

    def _parseClipping(self, clipping) -> None:
        """Set the min and max clipping values as tuple
        helper function to set the min and max values for clipping of the layer's weights.
        structure of an accepted tuple: (min, max)
        if tuple does not fit above structure, clipping will not be used.
        Also values in tuple must be numerics
        """
        if clipping is None:
            return
        if len(clipping) == 2 and clipping[0] < clipping[1]:
            try:
                float(clipping[0])
                float(clipping[1])
                self._clipping = clipping
            except ValueError:
                raise ValueError("Values must be numerics")
        else:
            raise ValueError("Clipping must be tuple of structure: (min, max)")

    @property
    def probability(self) -> float:
        return self._errorProbability

    @probability.setter
    def probability(self, probability: float) -> None:
        if probability < .0:
            raise ValueError("Probablility cannot be negative")
        self._errorProbability = probability

    @property
    def clipping(self) -> tuple | None:
        return self._clipping

    @clipping.setter
    def clipping(self, clipping: tuple) -> None:
        self._parseClipping(clipping)

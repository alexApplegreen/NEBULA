#!/usr/bin/env python3

"""
TrainingInjector.py
    injector to attach to untrained model to simulate errors during training
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from logging import Logger
from keras import Model, Layer

from NEBULA.utils.NoiseLayer import NoiseLayer
from NEBULA.utils.logging import getLogger


# TODO this is not trivial, implement this
class TrainingInjector:
    """Use to attach to untrained models
    This injector will simulate biterrors during the training phase
    """

    _logger: Logger
    _probability: float

    def __init__(self) -> None:
        self._logger = getLogger(__name__)

    def attach(self, model: Model) -> None:
        """Attach error injecting layer to existing untrained model
        This method will insert a layer into the given model to
        cause noise during backpropagation in the training of the model.
        """
        nl = NoiseLayer()(model.layers[-1].output)

#!/usr/bin/env python3

"""
legacyInjector.py:
    access to the WSA example functions using the injector-wrapper implementation
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from keras import Model, Layer

from NEBULA.core.baseInjector import BaseInjector
from NEBULA.core.legacy import flip_random_bits_in_model_weights
from NEBULA.utils.logging import getLogger


class LegacyInjector(BaseInjector):
    """Easy access to an error injector using the legacy implementation

    Attributes:
        _logger (Logger): Default logger
        _check (int): Used to secure the datatype in bit flip operations
        _layers (list[Layer]): Layers of the model under test
        _probability (float): Default Bit Error Rate
    """
    _logger = None
    _check = -1

    def __init__(self, layers: list[Layer], probability=0.01, check=-1) -> None:
        """Initializes a LegacyInjector instance

        Parameters:
            layers (list[Layer]): The layers of the model under test
            probability (float): Default Bit Error Rate
            check (int): Used to secure the datatype in bit flip operations
        """
        super().__init__(layers, probability)
        self._logger = getLogger(__name__)
        self._check = check

    def injectError(self, model: Model) -> None:
        """calls the og implementation from the WSA example
        This method edits the model inplace.

        Parameters:
            model (Model): The model under test
        """
        self._logger.debug(f"Injecting error with probability of {self._probability}")
        # edit model in place
        flip_random_bits_in_model_weights(model, self._probability, self._check)

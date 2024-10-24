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
    """
    _logger = None
    _check = -1

    def __init__(self, layers: list[Layer], probability=0.01, check=-1) -> None:
        super().__init__(layers, probability)
        self._logger = getLogger(__name__)
        self._check = check

    def injectError(self, model: Model) -> None:
        """calls the og implementation
        This method edits the model inplace.
        """
        self._logger.debug(f"Injecting error with probability of {self._probability}")
        # edit model in place
        flip_random_bits_in_model_weights(model, self._probability, self._check)

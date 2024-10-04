#!/usr/bin/env python

"""
logging.py:
    All logging related methods for API internal logging
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from keras import Model
from NEBULA.utils.logging import getLogger
from NEBULA.core.legacy import flip_random_bits_in_model_weights

class LegacyInjector():
    """Easy access to an errorinjector using the legacy implementation
    """
    _logger = None

    def __init__(self) -> None:
        self._logger = getLogger(__name__)

    def injectError(self, model : Model, probability : float = 0.001, check : int = -1) -> Model:
        """calls the og implementation"""
        return flip_random_bits_in_model_weights(model, probability, check)

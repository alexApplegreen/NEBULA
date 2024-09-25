#!/usr/bin/env python

"""
facade.py:
    Access to API functions which are visible to the user
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from NEBULA.utils.logging import getLogger
from NEBULA.core.legacy import flip_random_bits_in_model_weights, flip_single_number_float
class Facade():

    def __init__(self) -> None:
        self._logger = getLogger(__name__)

    def hello(self) -> str:
        return "Hello World"

    def legacy_flip_random_bits_in_model_weights(self, model, probability = 0.001, check=-1):
        """
        functions from provided script
        just passing through the arguments
        """
        return flip_random_bits_in_model_weights(model, probability, check)

    def legacy_flip_single_number_float(self, number_to_flip, data_type = 32, probability = 0.001, check = -1):
        """
        functions from provided script
        just passing through the arguments
        """
        return flip_single_number_float(number_to_flip, data_type, probability, check)

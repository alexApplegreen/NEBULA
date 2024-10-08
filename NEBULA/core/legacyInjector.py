#!/usr/bin/env python

"""
legacyInjector.py:
    access to the WSA example functions using the injector-wrapper implementation
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from logging import Logger

from keras import Model

from NEBULA.core.BaseInjector import BaseInjector
from NEBULA.utils.logging import getLogger
from NEBULA.core.legacy import flip_random_bits_in_model_weights


class LegacyInjector(BaseInjector):
    """Easy access to an error injector using the legacy implementation
    """
    _logger = None
    _check = -1

    def __init__(self, model: Model, probability=0.01, check=-1) -> None:
        super().__init__(model, probability)
        self._logger = getLogger(__name__)
        self._check = check

    def injectError(self) -> Model:
        """calls the og implementation and appends the new changed model to the history"""
        alteredModel = flip_random_bits_in_model_weights(self._model, self._probability, self._check)
        self._history.append(alteredModel)
        self._model = alteredModel
        return alteredModel

#!/usr/bin/env python

"""
logging.py:
    All logging related methods for API internal logging
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from keras import Model

from NEBULA.core.BaseInjector import BaseInjector
from NEBULA.utils.logging import getLogger


class Injector(BaseInjector):

    _logger = None

    def __init__(self, model : Model, probability : float = 0.01) -> None:
        super().__init__(model, probability)
        self._logger = getLogger(__name__)

    def injectError(self) -> Model:
        self._history.append(self._model)
        return self._model

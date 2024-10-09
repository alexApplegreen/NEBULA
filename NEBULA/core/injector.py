#!/usr/bin/env python3

"""
injector.py
    injector to use multithreading to inject errors
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from keras import Model
from keras.src.models.cloning import clone_model

from NEBULA.core.BaseInjector import BaseInjector
from NEBULA.core.injectionImpl import InjectionImpl
from NEBULA.utils.logging import getLogger


class Injector(BaseInjector):

    _logger = None

    def __init__(self, model: Model, probability: float = 0.01) -> None:
        super().__init__(model, probability)
        self._logger = getLogger(__name__)

    def injectError(self) -> Model:
        self._logger.debug(f"Injecting error with probability of {self._probability}")
        modelCopy = clone_model(self._model)
        modelCopy.set_weights(self._model.get_weights())
        modelCopy = InjectionImpl.injectToWeights(modelCopy, self._probability)
        self._history.append(self._model)
        return self._model

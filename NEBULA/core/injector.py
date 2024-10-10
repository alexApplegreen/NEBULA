#!/usr/bin/env python3

"""
injector.py
    injector to use multithreading to inject errors
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from tensorflow.keras import Model
from keras.src.models.cloning import clone_model

from NEBULA.core.BaseInjector import BaseInjector
from NEBULA.core.injectionImpl import InjectionImpl
from NEBULA.utils.logging import getLogger

import multiprocessing as mp


# TODO use reference to weights instead modifying the whole model

class Injector(BaseInjector):

    _logger = None
    _process_pool = None

    def __init__(self, model: Model, probability: float = 0.01) -> None:
        super().__init__(model, probability)
        self._logger = getLogger(__name__)
        self._process_pool = mp.Pool(len(model.layers))

    def __del__(self):
        self._process_pool.terminate()


    def injectError(self) -> Model:
        self._logger.debug(f"Injecting error with probability of {self._probability}")
        # create copy
        modelCopy = clone_model(self._model)
        modelCopy.set_weights(self._model.get_weights())

        # inject error
        modelCopy = InjectionImpl.injectToWeights(modelCopy, self._probability, self._process_pool)
        self._history.push(modelCopy)
        self._model = modelCopy
        return self._model

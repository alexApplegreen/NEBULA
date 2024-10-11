#!/usr/bin/env python3

"""
injector.py
    injector to use multiprocessing to inject errors
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
    """Class Injector:
    encapsulates all injection and other comfort functions towards
    modifying a model
    The injector will create a processpool at instantiation with one process
    per layer of the given model. These are used to inject errors into the model.
    This class also yields access to the history of changes made to the model through
    error injection
    """

    _logger = None
    _process_pool = None

    def __init__(self, model: Model, probability: float = 0.01) -> None:
        super().__init__(model, probability)
        self._logger = getLogger(__name__)
        self._process_pool = mp.Pool(len(model.layers))

    def __del__(self):
        self._process_pool.terminate()

    def injectError(self) -> Model:
        """ Method to inject errors into the model
        Creates a deep copy of the model and passes it to the
        injection implementation, which uses the processes from the pool
        to modify the model.
        Also adds the resulting model to the history
        """
        self._logger.debug(f"Injecting error with probability of {self._probability}")
        # create deep copy
        modelCopy = clone_model(self._model)
        modelCopy.set_weights(self._model.get_weights())

        # inject error
        modelCopy = InjectionImpl.injectToWeights(modelCopy, self._probability, self._process_pool)
        self._history.push(modelCopy)
        self._model = modelCopy
        return self._model

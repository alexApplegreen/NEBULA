#!/usr/bin/env python3

"""
injector.py
    injector to use multiprocessing to inject errors
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

import multiprocessing as mp
from logging import Logger
from multiprocessing import shared_memory

import numpy as np

from NEBULA.core.BaseInjector import BaseInjector
from NEBULA.core.injectionImpl import InjectionImpl
from NEBULA.utils.logging import getLogger


def _initialize_shared_weights(layers: dict) -> dict:
    """Initialize shared memory for each layer's weights."""
    shared_weights = {}
    for layer in layers:
        layer_name = layer.name
        shared_weights[layer_name] = {"weights": [], "membuf": []}

        for weight in layer.get_weights():
            shared_mem = shared_memory.SharedMemory(create=True, size=weight.nbytes)
            shared_weight = np.ndarray(weight.shape, dtype=weight.dtype, buffer=shared_mem.buf)
            np.copyto(shared_weight, weight)

            shared_weights[layer_name]["weights"].append(shared_weight)
            shared_weights[layer_name]["membuf"].append(shared_mem)

    return shared_weights


def _create_process_pool(layers: dict) -> mp.Pool:
    """Create a process pool with one process per layer."""
    num_processes = len(layers)
    return mp.Pool(num_processes)


class Injector(BaseInjector):
    """Class Injector:
    encapsulates all injection and other comfort functions towards
    modifying a model
    The injector will create a processpool at instantiation with one process
    per layer of the given model. These are used to inject errors into the model.
    This class also yields access to the history of changes made to the model through
    error injection
    """

    _logger: Logger
    _process_pool: mp.Pool
    _sharedWeights: dict

    def __init__(self, layers: dict, probability: float = 0.01) -> None:
        super().__init__(layers, probability)
        self._logger = getLogger(__name__)

        if mp.current_process().name == 'MainProcess':
            self._sharedWeights = _initialize_shared_weights(layers)
            self._process_pool = _create_process_pool(layers)

    def __del__(self):
        self._logger.debug("Closing Process Pool and deleting shared memory")
        if self._process_pool is not None:
            self._process_pool.terminate()
        for mem in self._sharedWeights:
            mem["membuf"].close()
            mem["membuf"].unlink()

    def injectError(self, model) -> None:
        """ Method to inject errors into the model
        Creates a deep copy of the model and passes it to the
        injection implementation, which uses the processes from the pool
        to modify the model.
        Also adds the resulting model to the history
        """
        self._logger.debug(f"Injecting error with probability of {self._probability}")

        # inject error
        InjectionImpl.injectToWeights(self._layers, self._probability, self._process_pool)
        # self._history.push(modelCopy)
        # self._model = modelCopy
        self._reconstructModel(model)

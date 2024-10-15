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
from keras import Model, Layer

from NEBULA.core.BaseInjector import BaseInjector
from NEBULA.core.injectionImpl import InjectionImpl
from NEBULA.utils.logging import getLogger


def _initialize_shared_weights(layers: list[Layer]) -> dict:
    """Initialize shared memory for each layer's weights."""
    shared_weights = {}
    for layer in layers:
        layer_name = layer.name
        shared_weights[layer_name] = {"membuf": [], "shapes": []}

        for weight in layer.get_weights():
            shared_mem = shared_memory.SharedMemory(create=True, size=weight.nbytes)
            shared_weight = np.ndarray(weight.shape, dtype=weight.dtype, buffer=shared_mem.buf)
            np.copyto(shared_weight, weight)

            shared_weights[layer_name]["membuf"].append(shared_mem)
            shared_weights[layer_name]["shapes"].append(weight.shape)

    return shared_weights


def _create_process_pool(layers: list[Layer]) -> mp.Pool:
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
    _process_pool: mp.Pool = None
    _sharedWeights: dict

    def __init__(self, layers: list[Layer], probability: float = 0.01) -> None:
        super().__init__(layers, probability)
        self._logger = getLogger(__name__)

        if mp.current_process().name == 'MainProcess':
            self._sharedWeights = _initialize_shared_weights(layers)
            self._process_pool = _create_process_pool(layers)

    def __del__(self):
        self._logger.debug("Closing Process Pool and deleting shared memory")
        if self._process_pool is not None:
            self._process_pool.terminate()
        for layer in self._sharedWeights:
            for i in range(len(self._sharedWeights[layer]["membuf"])):
                self._sharedWeights[layer]["membuf"][i].close()
                self._sharedWeights[layer]["membuf"][i].unlink()

    def injectError(self, model: Model) -> None:
        """ Method to inject errors into the model
        This method edits the model in place!
        Uses one process per layer of the given model and injects biterrors into the model
        with a Bit Error Rate of the given probability.
        """
        self._logger.debug(f"Injecting error with probability of {self._probability}")

        # inject error
        results = InjectionImpl.injectToWeights(self._sharedWeights, self._probability, self._process_pool)
        self._reconstructModel(model, results)
        self._history.push(model.layers)

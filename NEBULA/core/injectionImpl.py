#!/usr/bin/env python3

"""
InjectionImpl.py
    Actual modification of model weights is done here
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from threading import get_ident

import numpy as np

from NEBULA.utils.commons import flipAdjacentBits, flipFloat
from NEBULA.utils.logging import getLogger


class InjectionImpl:
    """Implementation of bit error injection to weights
    Since Tensorflow sets the GIL Lock for threads that reference model memory,
    this implementation uses processes

    Idea: Create deep copy of model before calling injection functions.
    Since processes operate on their own memory, one process per layer can modify
    the model's weights. When all processes are done, the model is written back
    """

    _logger = getLogger(__name__)

    @staticmethod
    def _concurrentErrorInjection(layername: str, layerMem: dict, probability: float):
        """Routine which is executed by the subprocesses
        The weights from the model's layer are read from shared memory
        and modified with a given probability and
        returns the modified weights.
        """
        InjectionImpl._logger.debug(
            f"started worker process {get_ident()} on layer {layername} with BER of {probability}"
        )

        weights = []

        try:
            weights = InjectionImpl._shmHelper(layerMem)
            newWeights = []
            for weight in weights:
                shape = weight.shape
                if weight.dtype == np.float32:
                    flattenedWeights = weight.flatten()
                    for i in range(len(flattenedWeights)):
                        flattenedWeights[i] = flipFloat(flattenedWeights[i], probability=probability)
                    newWeight = flattenedWeights.reshape(shape)
                    newWeights.append(newWeight)
                else:
                    newWeights.append(weight)
            return layername, newWeights

        except KeyError as e:
            # Shared Memory is not accessible, return unmodified weights
            InjectionImpl._logger.error(f"Cannot access argument {e.args[0]} of shared memory layer {layername}")
            return layername, weights

    @staticmethod
    def _concurrentStuckAtInjection(layername: str, layerMem: dict, probability: float):
        # TODO
        return layername, layerMem

    @staticmethod
    def _concurrentBurstInjection(layername: str, layerMem: dict, probability: float):
        InjectionImpl._logger.debug(
            f"started worker process {get_ident()} on layer {layername} with BER of {probability}"
        )

        weights = []

        try:
            weights = InjectionImpl._shmHelper(layerMem)
            newWeights = []

            for weight in weights:
                shape = weight.shape
                if weight.dtype == np.float32:
                    flattenedWeights = weight.flatten()
                    for i in range(len(flattenedWeights)):
                        flattenedWeights[i] = flipAdjacentBits(flattenedWeights[i], 3, probability)
                    newWeight = flattenedWeights.reshape(shape)
                    newWeights.append(newWeight)
                else:
                    newWeights.append(weight)

            return layername, newWeights

        except KeyError as e:
            # Shared Memory is not accessible, return unmodified weights
            InjectionImpl._logger.error(f"Cannot access argument {e.args[0]} of shared memory layer {layername}")
            return layername, weights


    @staticmethod
    def _shmHelper(layerMem: dict) -> list[np.ndarray]:
        weights = []
        for shm, shape in zip(layerMem["membuf"], layerMem["shapes"]):
            # TODO do not hardcode Float32!!!!
            weights.append(np.ndarray(shape, dtype=np.float32, buffer=shm.buf))
        return weights

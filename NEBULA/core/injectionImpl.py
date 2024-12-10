#!/usr/bin/env python3

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from threading import get_ident
from functools import wraps

import numpy as np

from NEBULA.utils.commons import flipAdjacentBits, flipFloat
from NEBULA.utils.logging import getLogger


def handleShmError(func):
    """Errorhandler that wraps annotated functions
    In case of errors when parsing the shared memory this wrapper will handle the error

    Parameters:
        func: The Function wrapped by the ErrorHandler
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            layername = kwargs.get("layername", "Unknown Layer")
            InjectionImpl._logger.error(f"Cannot access argument {e.args[0]} of shared memory layer {layername}")
            return layername, []
    return wrapper


class InjectionImpl:
    """Implementation of bit error injection to weights
    Since Tensorflow sets the GIL Lock for threads that reference model memory,
    this implementation uses processes

    Idea: Since processes operate on their own memory, one process per layer can modify
    the model's weights. When all processes are done, the model is written back

    This class is static

    Attributes:
        _logger (Logger) Default static logger
    """

    _logger = getLogger(__name__)

    @handleShmError
    @staticmethod
    def _concurrentErrorInjection(layername: str, layerMem: dict, probability: float) -> tuple:
        """Error injection routine for binomial distributed errors,
        which is executed by the subprocesses.
        The weights from the model's layer are read from shared memory
        and modified with a given probability and
        returns the modified weights.

        Parameters:
            layername (str): Name of the layer the error injection is applied to
            layerMem (dict): Mapping of layer names to shared memory buffers
            probability (float): Bit Error Rate in [0, 1]

        Returns:
            tuple: The layer name and an array of the modified weights
        """
        InjectionImpl._logger.debug(
            f"started worker process {get_ident()} on layer {layername} with BER of {probability}"
        )

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

    @staticmethod
    def _concurrentStuckAtInjection(layername: str, layerMem: dict, probability: float):
        """Error injection routine for stuck-at errors,
        which is executed by the subprocesses.
        The weights from the model's layer are read from shared memory
        and modified with a given probability and
        returns the modified weights.
        Stuck-at errors use the exact same implementation as usual binomial errors

        Parameters:
            layername (str): Name of the layer the error injection is applied to
            layerMem (dict): Mapping of layer names to shared memory buffers
            probability (float): Bit Error Rate in [0, 1]

        Returns:
            tuple: The layer name and an array of the modified weights
        """
        return InjectionImpl._concurrentErrorInjection(layername, layerMem, probability)

    @handleShmError
    @staticmethod
    def _concurrentBurstInjection(layername: str, layerMem: dict, probability: float):
        """Error injection routine for burst errors,
        which is executed by the subprocesses.
        The weights from the model's layer are read from shared memory
        and modified with a given probability and
        returns the modified weights.
        Burst errors flip a number of adjacent bits

        Parameters:
            layername (str): Name of the layer the error injection is applied to
            layerMem (dict): Mapping of layer names to shared memory buffers
            probability (float): Bit Error Rate in [0, 1]

        Returns:
            tuple: The layer name and an array of the modified weights
        """
        InjectionImpl._logger.debug(
            f"started worker process {get_ident()} on layer {layername} with BER of {probability}"
        )

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

    @staticmethod
    def _shmHelper(layerMem: dict) -> list[np.ndarray]:
        """Helper Function for concurrent injection routines
        Zips shared weight buffers and shapes for easy iteration
        Returns:
            list[np.array]: zipped shared memory buffers and shapes
        """
        weights = []
        for shm, shape in zip(layerMem["membuf"], layerMem["shapes"]):
            weights.append(np.ndarray(shape, dtype=np.float32, buffer=shm.buf))
        return weights

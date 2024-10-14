#!/usr/bin/env python3

"""
InjectionImpl.py
    Actual modification of model weights is done here
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

import multiprocessing as mp
from threading import get_ident

import numpy as np
import tensorflow as tf

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
    def injectToWeights(layers: dict, probability: float, processPool: mp.Pool) -> None:
        """Modify weights of model using multiprocessing.
        Tensorflow locks GIL which blocks all threads which are not tensorflow
        control flow. Processes can still run.
        Since python parameters are passed as object references, the dictionary is
        modified in place.
        """
        # Apply the error injection function to each layer in parallel
        _ = processPool.starmap_async(
            InjectionImpl._concurrentErrorInjection,
            [(layer, probability) for layer in layers]
        )

        # TODO are these necessary? starmap should block until processes return
        processPool.close()
        processPool.join()


    @staticmethod
    def _concurrentErrorInjection(layer, probability):
        """Routine which is executed by the subprocesses
        this function modifies the layer's weights with a given probability and
        returns the modified weights.
        """
        InjectionImpl._logger.info(
            f"started worker process {get_ident()} on layer {layer.name} with BER of {probability}"
        )

        newWeights = []
        for weight in layer.get_weights():
            if weight.dtype == np.float32:
                shape = weight.shape
                flattenedWeights = weight.flatten()
                for i in range(len(flattenedWeights)):
                    flattenedWeights[i] = InjectionImpl._flipFloat(flattenedWeights[i], probability=probability)
                newWeight = flattenedWeights.reshape(shape)
                newWeights.append(newWeight)
            else:
                newWeights.append(weight)
        return layer.name, newWeights

    @staticmethod
    def _flipFloat(number_to_flip, data_type=32, probability=0.001, check=-1):
        """Helper function which flips bits in a given memory range with a given probability
        returns the modified float number as a tf.tensor
        """
        random_numbers = np.random.rand(data_type + 1)
        flipped_bit_positions = np.where(random_numbers < probability)[0]
        if flipped_bit_positions.size == 0:
            return number_to_flip

        for pos in flipped_bit_positions:
            if data_type == 32:
                flip_mask = tf.bitwise.left_shift(tf.cast(1, tf.int32), pos)
                bitcast_to_int32 = tf.bitcast(number_to_flip, tf.int32)
                flipped_value = tf.bitwise.bitwise_xor(flip_mask, bitcast_to_int32)
                bitcast_to_float = tf.bitcast(flipped_value, tf.float32)
            elif data_type == 16:
                flip_mask = tf.bitwise.left_shift(tf.cast(1, tf.int16), pos)
                bitcast_to_int16 = tf.bitcast(number_to_flip, tf.int16)
                flipped_value = tf.bitwise.bitwise_xor(flip_mask, bitcast_to_int16)
                bitcast_to_float = tf.bitcast(flipped_value, tf.float16)
            else:
                print("data type ", data_type, " not valid")
            number_to_flip = bitcast_to_float

        if abs(bitcast_to_float) > check and check != -1:
            return 0
        else:
            return bitcast_to_float

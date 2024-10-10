import multiprocessing as mp
from threading import get_ident

import numpy as np
import tensorflow as tf
from keras.src.models import Model

from NEBULA.utils.logging import getLogger


class InjectionImpl:

    _logger = getLogger(__name__)

    @staticmethod
    def injectToWeights(model: Model, probability: float, processPool: mp.Pool) -> Model:
        """Modify weights of model using multiprocessing."""
        # Apply the error injection function to each layer in parallel
        results = processPool.starmap_async(
            InjectionImpl._concurrentErrorInjection,
            [(layer, probability) for layer in model.layers]
        )

        # TODO are these necessary?
        processPool.close()
        processPool.join()

        for result in results.get():
            layer_name, modified_weights = result
            layer = model.get_layer(name=layer_name)
            layer.set_weights(modified_weights)

        return model


    @staticmethod
    def _concurrentErrorInjection(layer, probability):
        InjectionImpl._logger.info(f"started worker process {get_ident()} on layer {layer.name} with BER of {probability}")

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

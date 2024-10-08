import numpy as np
import tensorflow as tf

from threading import Thread, get_ident
from keras.src.models import Model

from NEBULA.utils.logging import getLogger


class InjectionImpl:

    _logger = getLogger(__name__)

    @staticmethod
    def injectToWeights(model: Model, probability: float) -> Model:
        """modify weights of model in place using concurrency
        """
        threads = []
        for layer in model.layers:
            thread = Thread(
                target=InjectionImpl._concurrentErrorInjection,
                args=(layer, probability)
            )
            thread.start()

        for t in threads:
            t.join()

        return model

    @staticmethod
    def _concurrentErrorInjection(layer, probability):
        InjectionImpl._logger.debug(f"started worker thread {get_ident()} with BER of {probability}")

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
        layer.set_weights(newWeights)

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

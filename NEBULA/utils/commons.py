import struct
import tensorflow as tf
import numpy as np


def flipFloat(number_to_flip, data_type=32, probability=0.001, check=-1):
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


def loadFatModel():
    # TODO implement helper function to load model using NoiseLayer
    pass


def flipTensorBits(input: tf.Tensor, probability: float, dtype: np.dtype):
    if dtype is np.float32:
        x_bits = tf.bitcast(input, tf.int32)
        randomValues = tf.random.uniform(shape=tf.shape(x_bits), minval=0.0, maxval=1.0)
        flipMask = randomValues < probability
        bitPositions = tf.random.uniform(shape=tf.shape(x_bits), minval=0, maxval=32, dtype=tf.int32)
        bitFlips = tf.bitwise.left_shift(tf.ones_like(x_bits, dtype=tf.int32), bitPositions)

        flippedBits = tf.bitwise.bitwise_xor(x_bits, tf.where(flipMask, bitFlips, 0))
        flippedFloat = tf.bitcast(flippedBits, tf.float32)

        return flippedFloat
    else:
        return input


def binary(num):
    """
    Helper Function to display the binary representation of floating point numbers
    Source: https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
    Validated using https://www.h-schmidt.net/FloatConverter/IEEE754.html
    """
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

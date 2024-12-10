import tensorflow as tf

import numpy as np
import struct
import random


def flipFloat(
    number_to_flip: tf.Tensor,
    data_type: int = 32,
    probability: float = 0.001,
    check: int = -1
) -> tf.Tensor:
    """Helper function which flips bits in a given memory range with a given probability
    returns the modified float number as a tf.tensor

    Parameters:
        number_to_flip (float): Original value to modify
        data_type (int): Length of the memory word
        probability (float): Bit Error Rate
        check (int): Used to secure the input data type

    Returns:
        Tensor: modified value
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


def flipAdjacentBits(value: float, burstLength: int, probability: float) -> float:
    """
    Flips `n_bits` adjacent bits in a `float32` value at a random starting position,
    based on a given probability.

    Parameters:
        value (float): The original float32 value.
        burstLength (int): The number of adjacent bits to flip.
        probability (float): The probability (0 to 1) that the group of bits is flipped.

    Returns:
        float: The modified float32 value after flipping bits.
    """

    # Convert float32 to 32-bit binary representation
    packed = struct.pack('>f', value)
    int_value = struct.unpack('>I', packed)[0]

    # Convert to binary list of bits
    bit_list = list(f'{int_value:032b}')

    # Choose a random starting index for adjacent bits
    start_index = random.randint(0, 32 - burstLength)

    # Flip adjacent bits based on the probability
    if random.random() < probability:
        for i in range(start_index, start_index + burstLength):
            bit_list[i] = '1' if bit_list[i] == '0' else '0'

    # Convert back to integer
    flipped_int_value = int("".join(bit_list), 2)

    # Convert back to float32
    flipped_packed = struct.pack('>I', flipped_int_value)
    flipped_value = struct.unpack('>f', flipped_packed)[0]

    return flipped_value


def flipTensorBits(input: tf.Tensor, probability: float, dtype: np.dtype) -> tf.Tensor:
    """New implementation of legacy flipFloat implementation
    Binomially distributed random bit flips with given probability of exactly 1 bit

    Parameters:
        input (Tensor): input value to modify
        probability (float): Bit Error Rate
        dtype (dtype): Datatype of the input value

    Returns:
        Tensor: The modified value as a Tensor
    """
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

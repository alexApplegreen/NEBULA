#!/usr/bin/env python3

"""
legacy.py:
    Original bit flip operations from the WSA example
    These implementations were left in to allow for benchmarking
"""

__author__      = "Tim Düe"

import numpy as np
import tensorflow as tf

def flip_single_number_float(number_to_flip, data_type = 32, probability = 0.001, check = -1):
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

def flip_random_bits_in_model_weights(model, probability = 0.001, check=-1):
    for layer in model.layers:
        layer_weights = layer.get_weights()
        new_weights = []
        layer_idx = 1
        for weight_tensor in layer_weights:
            print("Working on layer ", layer_idx, " out of ", len(layer_weights))
            layer_idx = layer_idx + 1
            if weight_tensor.dtype == np.float32:
                shape = weight_tensor.shape
                flattened_weights = weight_tensor.flatten()
                for i in range(len(flattened_weights)):
                    flattened_weights[i] = flip_single_number_float(flattened_weights[i], probability=probability, check=check)
                new_weight_tensor = flattened_weights.reshape(shape)
                new_weights.append(new_weight_tensor)
            else:
                new_weights.append(weight_tensor)
        layer.set_weights(new_weights)
    return model

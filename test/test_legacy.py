import unittest

import keras
import numpy as np
from tensorflow import float32, Variable

from NEBULA.core.legacy import flip_random_bits_in_model_weights, flip_single_number_float


class LegacyTest(unittest.TestCase):

    def test_flipSingleNumberFloat(self):
        number = Variable(initial_value=123.0, dtype=float32)
        numberAltered = flip_single_number_float(number, probability=1.0)
        self.assertNotEqual(number, numberAltered)


    def test_flipRandomBitsInModelWeights(self):
        inputs = keras.Input(shape=(37,))
        x = keras.layers.Dense(32, activation="relu")(inputs)
        outputs = keras.layers.Dense(5, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        modelAltered = flip_random_bits_in_model_weights(model, probability=1.0)
        changed = False
        for new, orig in zip(modelAltered.get_weights(), model.get_weights()):
            changed = np.allclose(orig, new)
            if changed:
                break

        self.assertTrue(changed)

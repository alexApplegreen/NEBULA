import unittest

import numpy as np
import tensorflow as tf
from tensorflow._api.v2.errors import InvalidArgumentError

from NEBULA.utils.noiseLayer import NoiseLayer


class NoiseLayerTest(unittest.TestCase):

    def test_callShouldInjectErrors(self):
        with self.assertRaises(InvalidArgumentError):
            inputs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            inputs = tf.constant(inputs)
            nl = NoiseLayer(probability=1.0)
            corruptedInputs = nl.call(inputs, training=True)
            tf.debugging.assert_equal(inputs, corruptedInputs)

    def test_callShouldNotInjectDuringInference(self):
        try:
            inputs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            inputs = tf.constant(inputs)
            nl = NoiseLayer(probability=1.0)
            corruptedInputs = nl.call(inputs, training=False)
            tf.debugging.assert_equal(inputs, corruptedInputs)
        except InvalidArgumentError:
            self.fail()

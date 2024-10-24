import unittest

import numpy as np
import tensorflow as tf
from tensorflow._api.v2.errors import InvalidArgumentError

from NEBULA.core.noiseLayer import NoiseLayer


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

    def test_invalidClippingParseDiesOnInvalidTuple(self):
        with self.assertRaises(ValueError):
            _ = NoiseLayer(probability=0.0, clipping=(100, 0))

    def test_wrongTypeInClippingRaisesError(self):
        with self.assertRaises(ValueError):
            _ = NoiseLayer(clipping=("string", "NotANumber"))

    def test_setClippingTupleWorks(self):
        nl = NoiseLayer(probability=0.5, clipping=(-1, 1))
        self.assertEqual(nl.probability, 0.5)
        self.assertEqual(nl.clipping, (-1, 1))

    def test_clippingIsUsed(self):
        inputs = np.array([10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=np.float32)
        inputs = tf.constant(inputs)

        nl = NoiseLayer(probability=0.0, clipping=(-1.0, 1.0))
        res = nl.call(inputs, training=True)
        for item in res.numpy():
            self.assertGreaterEqual(item, -1.0)
            self.assertLessEqual(item, 1.0)

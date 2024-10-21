import unittest

import numpy as np

from NEBULA.utils.noiseLayer import NoiseLayer


class NoiseLayerTest(unittest.TestCase):

    def test_callShouldInjectErrors(self):
        inputs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        nl = NoiseLayer(probability=1.0)
        corruptedInputs = nl.call(inputs, training=True)

        for orig, new in zip(inputs, corruptedInputs):
            self.assertFalse(np.allclose(orig, new))

    def test_callShouldNotInjectDuringInference(self):
        inputs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        nl = NoiseLayer(probability=1.0)
        corruptedInputs = nl.call(inputs, training=False)

        for orig, new in zip(inputs, corruptedInputs):
            self.assertTrue(np.allclose(orig, new))

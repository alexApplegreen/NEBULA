import unittest

import keras
import numpy as np

from NEBULA.core.legacyInjector import LegacyInjector


class LegacyInjectorTest(unittest.TestCase):

    _model = None

    def setUp(self):
        if self._model is None:
            inputs = keras.Input(shape=(37,))
            x = keras.layers.Dense(32, activation="relu")(inputs)
            outputs = keras.layers.Dense(5, activation="softmax")(x)
            self._model = keras.Model(inputs=inputs, outputs=outputs)

    def test_injectErrorWith0ProbabilityDoesNotChangeModel(self):
        li = LegacyInjector(self._model.layers, probability=0.0)
        weightsOld = self._model.get_weights()
        li.injectError(self._model)
        weightsNew = self._model.get_weights()
        for orig, new in zip(weightsNew, weightsOld):
            self.assertTrue(np.allclose(orig, new))

    def test_injectErrorDoesChangeModel(self):
        li = LegacyInjector(self._model.layers, probability=1.0)
        weightsOld = self._model.get_weights()
        li.injectError(self._model)
        weightsNew = self._model.get_weights()
        allSame = True
        for orig, new in zip(weightsOld, weightsNew):
            allSame = np.allclose(orig, new)
            if not allSame:
                break

        self.assertFalse(allSame)
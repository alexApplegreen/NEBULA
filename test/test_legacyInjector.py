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
        li = LegacyInjector(self._model)
        li.probability = 0.0
        modelAltered = li.injectError()
        weightsOld = self._model.get_weights()
        weightsNew = modelAltered.get_weights()
        for orig, new in zip(weightsNew, weightsOld):
            self.assertTrue(np.allclose(orig, new))

    def test_injectErrorDoesChangeModel(self):
        li = LegacyInjector(self._model)
        li.probability = 1.0
        modelAltered = li.injectError()
        weightsOld = self._model.get_weights()
        weightsNew = modelAltered.get_weights()
        changed = False
        for orig, new in zip(weightsNew, weightsOld):
            changed = np.allclose(orig, new)
            if changed:
                break

        self.assertTrue(changed)

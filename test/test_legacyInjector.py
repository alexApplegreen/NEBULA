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
        li = LegacyInjector(self._model, probability=0.0)
        weightsOld = li.model.get_weights()
        li.injectError()
        weightsNew = li.model.get_weights()
        for orig, new in zip(weightsNew, weightsOld):
            self.assertTrue(np.allclose(orig, new))

    def test_injectErrorDoesChangeModel(self):
        li = LegacyInjector(self._model, probability=1.0)
        weightsOld = li.model.get_weights()
        modelAltered = li.injectError()
        weightsNew = modelAltered.get_weights()
        allSame = True
        for orig, new in zip(weightsOld, weightsNew):
            allSame = np.allclose(orig, new)
            if not allSame:
                break

        self.assertFalse(allSame)

    def test_changeIsReversibleWithActualModel(self):
        li = LegacyInjector(self._model, probability=1.0)
        origWeights = self._model.get_weights()
        _ = li.injectError()

        undoneModel = li.undo()
        undoneWeights = undoneModel.get_weights()

        for orig, new in zip(origWeights, undoneWeights):
            self.assertTrue(np.allclose(orig, new))


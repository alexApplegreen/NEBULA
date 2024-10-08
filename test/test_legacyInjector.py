import unittest
from unittest.mock import Mock

import keras
import numpy as np

from NEBULA.core.legacyInjector import LegacyInjector


class LegacyInjectorTest(unittest.TestCase):

    _model = None

    def setUp(self):
        # Create a mock Keras model
        self.mock_model = Mock()
        self.mock_model2 = Mock()
        self.mock_model3 = Mock()
        # Set up the mock model's behavior
        self.mock_model.get_weights.return_value = [[1, 2], [3, 4]]
        self.mock_model2.get_weights.return_value = [[1, 1], [3, 4]]
        self.mock_model3.get_weights.return_value = [[1, 2], [1, 4]]

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

    def test_changeIsReversible(self):
        # create injector from fresh model
        li = LegacyInjector(self.mock_model, probability=1.0)
        origWeights = li.model.get_weights()

        # mock error injection
        li.model = self.mock_model2
        li._history.push(self.mock_model2)

        li.undo()
        undoneWeights = li.model.get_weights()

        for orig, new in zip(origWeights, undoneWeights):
            self.assertTrue(np.allclose(orig, new))

    def test_changeIsReversibleWithActualModel(self):
        # TODO this stuff is not working with real models?????
        li = LegacyInjector(self._model, probability=1.0)
        origWeights = self._model.get_weights()
        val = li._history[0].get_weights()
        alteredModel = li.injectError()
        val2 = li._history[-1].get_weights()
        alteredWeights = alteredModel.get_weights()

        undoneModel = li.undo()
        undoneWeights = undoneModel.get_weights()

        for orig, new in zip(origWeights, undoneWeights):
            self.assertTrue(np.allclose(orig, new))


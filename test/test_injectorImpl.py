import unittest
from unittest.mock import Mock

import keras
from keras.src.models.cloning import clone_model
import numpy as np

from NEBULA.core.injectionImpl import InjectionImpl


class TestInjectorImpl(unittest.TestCase):

    _model = None
    mockLayer = None

    def setUp(self):
        if self._model is None:
            inputs = keras.Input(shape=(37,))
            x = keras.layers.Dense(32, activation="relu")(inputs)
            outputs = keras.layers.Dense(5, activation="softmax")(x)
            self._model = keras.Model(inputs=inputs, outputs=outputs)

    def test_ConcurrentRoutine(self):
        modelCopy = clone_model(self._model)
        modelCopy.set_weights(self._model.get_weights())
        origWeights = self._model.get_weights()

        InjectionImpl._concurrentErrorInjection(modelCopy.layers[1], probability=1.0)
        newWeights = modelCopy.get_weights()

        allSame = True
        for orig, new in zip(origWeights, newWeights):
            allSame = np.allclose(orig, new)
            if not allSame:
                break
        self.assertFalse(allSame)

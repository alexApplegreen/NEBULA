import unittest
import keras
import numpy as np
from NEBULA.core.injector import Injector

class InjectorTest(unittest.TestCase):

    _model = None

    def setUp(self):
        if self._model is None:
            inputs = keras.Input(shape=(37,))
            x = keras.layers.Dense(32, activation="relu")(inputs)
            outputs = keras.layers.Dense(5, activation="softmax")(x)
            self._model = keras.Model(inputs=inputs, outputs=outputs)

    def test_injectorWithZeroProbDoesNotChangeModel(self):
        injector = Injector(self._model, 1.0)
        weightsOrig = injector.model.get_weights()
        _ = injector.injectError()
        weightsNew = injector.model.get_weights()

        for orig, new in zip(weightsOrig, weightsNew):
            self.assertTrue(np.allclose(orig, new))

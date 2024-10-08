import unittest
import keras
from NEBULA.core.legacyInjector import LegacyInjector


class LegacyInjectorTest(unittest.TestCase):

    def test_injectErrorWith0ProbabilityDoesNotChangeModel(self):
        inputs = keras.Input(shape=(37,))
        x = keras.layers.Dense(32, activation="relu")(inputs)
        outputs = keras.layers.Dense(5, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        li = LegacyInjector(model)
        li.probability = 0.0
        modelAlterted = li.injectError()
        self.assertEqual(model, modelAlterted)

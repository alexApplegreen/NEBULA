import unittest

from keras import Model
import numpy as np

from NEBULA.core.injector import Injector
from NEBULA.utils.logging import getLogger
from utils import ModelUtils

import tensorflow.keras


class InjectorTest(unittest.TestCase):

    _model = None
    _logger = getLogger(__name__)

    def setUp(self):
        if self._model is None:
            self._model = ModelUtils.ModelUtils.getBasicModel()

    def test_injectorWithHundredProbChangesModel(self):
        weightsOrig = self._model.get_weights()
        injector = Injector(self._model.layers, 1.0)
        injector.injectError(self._model)
        weightsNew = self._model.get_weights()

        allSame = True
        for orig, new in zip(weightsOrig, weightsNew):
            allSame = np.allclose(orig, new)
            if not allSame:
                break
        self.assertFalse(allSame)

    def test_reconstructModelWorks(self):
        modelCopy = ModelUtils.ModelUtils.getBasicModel()

        # overwrite copy weights with zeros
        for idx, layer in enumerate(self._model.layers):
            original_weights = layer.get_weights()
            new_weights = [np.zeros(w.shape) for w in original_weights]
            copyLayer = modelCopy.get_layer(index=idx)
            copyLayer.set_weights(new_weights)

        # bring layer objects into dict form
        layers = list()
        for idx, layer in enumerate(modelCopy.layers):
            # Get the name of the layer
            layer_name = layer.name
            layer_weights = self._model.get_layer(index=idx).get_weights()
            # Store the weights in the dictionary
            layers.append((layer_name, layer_weights))

        # self._logger.debug(layers)

        Injector._reconstructModel(modelCopy, layers)

        for orig, new in zip(self._model.get_weights(), modelCopy.get_weights()):
            self.assertTrue(np.allclose(orig, new))

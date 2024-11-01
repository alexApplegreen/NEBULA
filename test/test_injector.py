import copy
import unittest

from keras import Model
import numpy as np

from NEBULA.core.injector import Injector
from NEBULA.utils.logging import getLogger
from NEBULA.core.errorTypes import ErrorTypes
from utils.ModelUtils import ModelUtils

import tensorflow.keras


class InjectorTest(unittest.TestCase):

    _model = None
    _logger = getLogger(__name__)

    def setUp(self):
        if self._model is None:
            self._model = ModelUtils.getBasicModel()

    def test_injectorWithHundredProbChangesModel(self):
        weightsOrig = self._model.get_weights()
        injector = Injector(self._model.layers, probability=1.0)
        injector.injectError(self._model)
        weightsNew = self._model.get_weights()

        allSame = True
        for orig, new in zip(weightsOrig, weightsNew):
            allSame = np.allclose(orig, new)
            if not allSame:
                break
        self.assertFalse(allSame)

    def test_reconstructModelWorks(self):
        injector = Injector(self._model.layers)
        modelCopy = ModelUtils.getBasicModel()

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

        injector._reconstructModel(modelCopy, layers)

        for orig, new in zip(self._model.get_weights(), modelCopy.get_weights()):
            self.assertTrue(np.allclose(orig, new))

    def test_processPoolCanExecuteMultipleTasks(self):
        injector = Injector(self._model.layers, probability=0.000005)
        try:
            injector.injectError(self._model)
            injector.injectError(self._model)
        except ValueError:
            # ValueError is thrown when task is queued before last injection is done
            self.fail()


    def test_changesAreReversible(self):
        weightsOrig = self._model.get_weights()

        #inject error
        injector = Injector(self._model.layers, probability=1.0)
        injector.injectError(self._model)

        weightsChanged = self._model.get_weights()

        allSame = True
        for orig, new in zip(weightsOrig, weightsChanged):
            allSame = np.allclose(orig, new)
            if not allSame:
                break
        self.assertFalse(allSame)

        injector.undo(self._model)

        weightsUndone = self._model.get_weights()

        for orig, new in zip(weightsOrig, weightsUndone):
            self.assertTrue(np.allclose(orig, new))

    def test_undoOnEmptyHistoryShouldRaiseIndexError(self):
        with self.assertRaises(IndexError):
            injector = Injector(self._model.layers)
            injector.undo(self._model)

    def test_undoOnWrongModelShouldRaiseAttributeError(self):
        with self.assertRaises(ValueError):
            injector = Injector(self._model.layers, probability=.0)
            otherModel = ModelUtils.getBasicModel()
            injector.injectError(self._model)
            injector.undo(otherModel)

    def test_updatingAfterInjectionWorks(self):
        injector = Injector(self._model.layers, probability=1.0)

        for layer in self._model.layers:
            data = injector._sharedWeights[layer.name]
            for idx, weight in enumerate(layer.get_weights()):
                weightFromBuffer = np.ndarray(
                    data['shapes'][idx],
                    dtype=np.float32,
                    buffer=data['membuf'][idx].buf
                )
                if np.any(np.isnan(weight)):
                    self.assertEqual(len(weight), len(weightFromBuffer))
                else:
                    self.assertTrue(np.allclose(weight, weightFromBuffer))

    def test_injectorWorksWithBurstErrorType(self):
        weightsOrig = self._model.get_weights()
        injector = Injector(self._model.layers, probability=1.0)
        injector.injectError(self._model, errorType=ErrorTypes.BURST)
        weightsNew = self._model.get_weights()

        allSame = True
        for orig, new in zip(weightsOrig, weightsNew):
            allSame = np.allclose(orig, new)
            if not allSame:
                break
        self.assertFalse(allSame)

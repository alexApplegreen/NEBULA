import unittest

import numpy as np

from NEBULA.core.injector import Injector
from utils import ModelUtils


class InjectorTest(unittest.TestCase):

    _model = None

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

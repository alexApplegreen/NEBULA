import unittest
from multiprocessing import shared_memory
from unittest.mock import Mock

import numpy as np

from NEBULA.core.injectionImpl import InjectionImpl


class TestInjectorImpl(unittest.TestCase):

    _model = None
    _layerMem = {
        "membuf": [None],
        "shapes": [(2,)]
    }

    def setUp(self):
        self._model = Mock()
        self._model.get_weights.return_value = [1, 2]
        data = np.array([1, 2])
        self.shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        sharedData = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm.buf)
        np.copyto(sharedData, data)
        self._layerMem["membuf"][0] = self.shm



    def test_ConcurrentRoutine(self):
        origWeights = self._model.get_weights()

        layerName, newWeights = InjectionImpl._concurrentErrorInjection("Test", self._layerMem, probability=1.0)

        self.assertEqual("Test", layerName)
        self.assertNotEqual(origWeights, newWeights)

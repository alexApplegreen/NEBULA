import unittest
from multiprocessing import shared_memory
from unittest.mock import Mock

import numpy as np

from NEBULA.core.injectionImpl import InjectionImpl
from NEBULA.utils.logging import getLogger


class TestInjectorImpl(unittest.TestCase):

    _model = None
    _layerMem = {
        "membuf": [None],
        "shapes": [(2,)]
    }
    _logger = getLogger(__name__)

    def setUp(self):
        self._model = Mock()
        self._model.get_weights.return_value = [1, 2]
        data = np.array([1, 2])
        self.shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        sharedData = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm.buf)
        np.copyto(sharedData, data)
        self._layerMem["membuf"][0] = self.shm

    def tearDown(self):
        try:
            self._layerMem["membuf"][0].close()
            self._layerMem["membuf"][0].unlink()
        except KeyError:
            self._logger.warn("Could not clean up shared memory from unittest")

    def test_ConcurrentRoutine(self):
        origWeights = self._model.get_weights()

        layerName, newWeights = InjectionImpl._concurrentErrorInjection("Test", self._layerMem, probability=1.0)

        self.assertEqual("Test", layerName)
        self.assertNotEqual(origWeights, newWeights)

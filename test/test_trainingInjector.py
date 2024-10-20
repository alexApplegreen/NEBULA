import unittest

from NEBULA.core.TrainingInjector import TrainingInjector
from NEBULA.core.TrainingInjector import _buildFunctionalModel, _buildSequentialModel
from NEBULA.utils.NoiseLayer import NoiseLayer
from utils.ModelUtils import ModelUtils


class TrainingInjectorTest(unittest.TestCase):

    def test_buildFuncModelWorks(self):
        model = ModelUtils.getBasicModel()
        nl = NoiseLayer()
        model = _buildFunctionalModel(model, nl, 2)
        self.assertEqual(len(model.layers), 4)
        self.assertTrue("noise_layer" in model.get_layer(index=2).name)

    def test_buildSeqModelWorks(self):
        model = ModelUtils.getSequentialModel()
        nl = NoiseLayer()
        model = _buildSequentialModel(model, nl, 2)
        self.assertEqual(len(model.layers), 4)
        self.assertTrue("noise_layer" in model.get_layer(index=2).name)

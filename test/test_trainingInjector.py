import unittest

from NEBULA.core.trainingInjector import TrainingInjector
from NEBULA.core.trainingInjector import _buildFunctionalModel, _buildSequentialModel
from NEBULA.utils.noiseLayer import NoiseLayer
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

    def test_attachFailsWithNegativeIndex(self):
        with self.assertRaises(ValueError):
            ti = TrainingInjector()
            model = ModelUtils.getBasicModel()
            ti.attach(model, -1)

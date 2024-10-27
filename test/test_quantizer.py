import unittest

from NEBULA.core.quantizer import Quantizer
from utils.ModelUtils import ModelUtils

class QuantizerTest(unittest.TestCase):

    def test_quantizeDoesNotAlterParam(self):
        model = ModelUtils.getBasicModel()
        q = Quantizer()

        quantModel = q.quantize(model)

        self.assertFalse(id(model) == id(quantModel))

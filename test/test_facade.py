import unittest
from NEBULA.facade import Facade

class MainTest(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._facade = Facade()

    def test_main(self):
        retVal = self._facade.hello()
        self.assertEqual(retVal, "Hello World")

    def test_flipBits(self):
        # TODO
        self.assertTrue(True)

    def test_flipWeightsInModel(self):
        # TODO
        self.assertTrue(True)

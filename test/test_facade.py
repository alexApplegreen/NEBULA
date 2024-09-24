import unittest
from NEBULA.facade import Facade

class MainTest(unittest.TestCase):

    def test_main(self):
        facade = Facade()
        retVal = facade.hello()
        self.assertEqual(retVal, "Hello World")

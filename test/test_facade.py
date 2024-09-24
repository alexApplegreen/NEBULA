import unittest
from src.facade import Facade

class MainTest(unittest.TestCase):

    def test_main(self):
        facade = Facade()
        retVal = facade.hello()
        self.assertEqual(retVal, "Hello World")

if __name__ == "__main__":
    unittest.main()

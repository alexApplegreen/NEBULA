import unittest
from src.main import hello

class MainTest(unittest.TestCase):

    def test_main(self):
        self.assertEqual(hello(), "Hello World")

if __name__ == "__main__":
    unittest.main()

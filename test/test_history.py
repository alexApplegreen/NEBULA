import unittest
from unittest.mock import Mock

from NEBULA.core.history import History


class HistoryTest(unittest.TestCase):

    def setUp(self):
        # Create a mock Keras model
        self.mockLayer1 = Mock()
        self.mockLayer2 = Mock()
        self.mockLayer3 = Mock()
        # Set up the mock model's behavior
        self.mockLayer1.get_weights.return_value = 1
        self.mockLayer2.get_weights.return_value = 2
        self.mockLayer3.get_weights.return_value = 3

    def test_pushWorks(self):
        history = History()
        history.push(self.mockLayer1)
        result = history.pop()
        self.assertEqual(1, result.get_weights())

    def test_popWorks(self):
        history = History()
        history.push(self.mockLayer1)
        size1 = history.size()
        _ = history.pop()
        size2 = history.size()
        self.assertNotEqual(size1, size2)

    def test_isFifo(self):
        history = History()
        history.push(self.mockLayer1)
        history.push(self.mockLayer2)
        history.push(self.mockLayer3)

        first = history.pop()
        second = history.pop()
        third = history.pop()

        self.assertEqual(first.get_weights(), 3)
        self.assertEqual(second.get_weights(), 2)
        self.assertEqual(third.get_weights(), 1)

    def test_peekDoesNotDelete(self):
        history = History()
        history.push(self.mockLayer1)
        size1 = history.size()
        _ = history.peek()
        size2 = history.size()
        self.assertEqual(size1, size2)

    def test_revertWorks(self):
        history = History()
        history.push(self.mockLayer1)
        history.push(self.mockLayer2)
        layersAltered = history.peek()
        self.assertEqual(layersAltered.get_weights(), 2)
        history.revert()
        layers = history.peek()
        self.assertEqual(layers.get_weights(), 1)

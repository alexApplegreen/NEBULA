import unittest
from unittest.mock import Mock

from NEBULA.core.history import History


class HistoryTest(unittest.TestCase):

    def setUp(self):
        # Create a mock Keras model
        self.mock_model = Mock()
        self.mock_model2 = Mock()
        self.mock_model3 = Mock()
        # Set up the mock model's behavior
        self.mock_model.get_weights.return_value = [[1, 2], [3, 4]]
        self.mock_model2.get_weights.return_value = [[1, 1], [3, 4]]
        self.mock_model3.get_weights.return_value = [[1, 2], [1, 4]]

    def test_pushWorks(self):
        history = History()
        history.push(self.mock_model)
        model = history.pop()
        self.assertEqual(model.get_weights(), [[1, 2], [3, 4]])

    def test_popWorks(self):
        history = History()
        history.push(self.mock_model)
        size1 = history.size()
        _ = history.pop()
        size2 = history.size()
        self.assertNotEqual(size1, size2)

    def test_isFifo(self):
        history = History()
        history.push(self.mock_model)
        history.push(self.mock_model2)
        history.push(self.mock_model3)

        first = history.pop()
        second = history.pop()
        third = history.pop()

        self.assertEqual(first.get_weights(), self.mock_model3.get_weights())
        self.assertEqual(second.get_weights(), self.mock_model2.get_weights())
        self.assertEqual(third.get_weights(), self.mock_model.get_weights())

    def test_peekDoesNotDelete(self):
        history = History()
        history.push(self.mock_model)
        size1 = history.size()
        _ = history.peek()
        size2 = history.size()
        self.assertEqual(size1, size2)

    def test_revertWorks(self):
        history = History()
        history.push(self.mock_model)
        history.push(self.mock_model2)
        modelAltered = history.peek()
        self.assertEqual(modelAltered.get_weights(), [[1, 1], [3, 4]])
        history.revert()
        model = history.peek()
        self.assertEqual(model.get_weights(), [[1, 2], [3, 4]])

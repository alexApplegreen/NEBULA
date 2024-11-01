import unittest
import tensorflow as tf
import numpy as np

from tensorflow._api.v2.errors import InvalidArgumentError

from NEBULA.utils.commons import flipAdjacentBits, flipTensorBits
from NEBULA.utils.helper import binary


class CommonsTest(unittest.TestCase):

    def test_flipTensorBits(self):
        with self.assertRaises(InvalidArgumentError):
            tensor = tf.constant([1.0], dtype=tf.float32)
            tensorFlipped = flipTensorBits(tensor, probability=1.0, dtype=np.float32)
            tf.debugging.assert_equal(tensor, tensorFlipped)

    def test_burstErrorsAreActuallyAdjacent(self):
        number = 12.0
        binString = binary(number)
        numFlipped = flipAdjacentBits(number, 3, 1.0)
        binStringFlipped = binary(numFlipped)
        self.assertEqual(len(binString), len(binStringFlipped))
        diff = list()
        for i in range(len(binString)):
            if binString[i] != binStringFlipped[i]:
                diff.append(i)
        self.assertEqual(len(diff), 3)
        self.assertEqual(diff[-1] - diff[0], 2)

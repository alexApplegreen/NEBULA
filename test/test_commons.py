import unittest
import struct

from tensorflow._api.v2.errors import InvalidArgumentError

from NEBULA.utils.commons import flipTensorBits

import tensorflow as tf
import numpy as np

class CommonsTest(unittest.TestCase):

    def test_flipTensorBits(self):
        with self.assertRaises(InvalidArgumentError):
            tensor = tf.constant([1.0], dtype=tf.float32)
            tensorFlipped = flipTensorBits(tensor, probability=1.0, dtype=np.float32)
            tf.debugging.assert_equal(tensor, tensorFlipped)

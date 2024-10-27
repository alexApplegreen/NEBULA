from logging import Logger

from keras.src.models.cloning import clone_model
from NEBULA.utils import getLogger

import tensorflow as tf
from keras import Model

class Quantizer():
    """Class for performing quantization on keras models.
    This class has functions for 2 different kinds of quantization:
        1. quantization aware training
        2. post training quantization
    """

    _logger: Logger

    def __init__(self) -> None:
        self._logger = getLogger(__name__)

    def quantize(self, model: Model) -> Model:
        # TODO add datatype parameter
        """Perform post training quantization on a model
        quantizes all weights of a model to a given datatype.
        Returns a new instance of keras.model
        The original model is not modified
        """
        # TODO actually quantize model
        quantModel = clone_model(model)
        return quantModel

    def quantizeAwareTrain(self, model: Model)-> None:
        # TODO do something clever here
        pass

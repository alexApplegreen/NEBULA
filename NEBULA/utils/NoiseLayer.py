from logging import Logger
from keras.src import Layer

from NEBULA.utils.logging import getLogger


class NoiseLayer(Layer):

    _logger: Logger

    def __init__(self, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            dtype=dtype,
            autocast=autocast,
            name=name,
            **kwargs
        )
        if not trainable:
            raise ValueError("Gaussian Noise Layer cannot be created from untrainable layer")
        self._logger = getLogger(__name__)

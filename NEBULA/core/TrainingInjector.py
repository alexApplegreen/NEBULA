#!/usr/bin/env python3

"""
TrainingInjector.py
    injector to attach to untrained model to simulate errors during training
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from logging import Logger

from keras import Model, Layer
from keras.src.models import Functional, Sequential

from NEBULA.utils.noiseLayer import NoiseLayer
from NEBULA.utils.logging import getLogger


def _buildSequentialModel(model: Model, noiseLayer: Layer, index: int) -> Model:
    layers = model.layers
    pre_layers = layers[:index]  # layers before the insertion point
    post_layers = layers[index:]  # layers after the insertion point

    newModel = Sequential()

    for layer in pre_layers:
        newModel.add(layer)
    newModel.add(noiseLayer)
    for layer in post_layers:
        newModel.add(layer)

    return newModel


def _buildFunctionalModel(model: Model, noiseLayer: Layer, index: int) -> Model:
    inputs = model.input
    x = inputs

    for i, layer in enumerate(model.layers):
        if i == index:
            x = noiseLayer(x)
        if i != 0:
            x = layer(x)

    return Functional(inputs=inputs, outputs=x)


class TrainingInjector():
    """Use to attach to untrained models
    This injector will simulate biterrors during the training phase
    by adding a noiselayer to the given model.
    This layer will produce noise during the learning phase of the network.
    During inference the layer will pass all values from the n-1th layer through
    """

    _logger: Logger
    _probability: float

    def __init__(self, probability: float = 0.01) -> None:
        if probability < .0:
            raise ValueError("BER cannot be negative")
        self._logger = getLogger(__name__)
        self._probability = probability

    def attach(self, model: Model, index: int = 1) -> Model:
        if index < 0:
            raise ValueError("Index cannot be negative")
        # TODO allow user to pass in Noiselayer instance or parameters for it
        """Attach error injecting layer to existing untrained model
        This method will insert a layer into the given model to
        cause noise during backpropagation in the training of the model.
        index parameter specifies where in the model to attach the pertubating layer.
        The placement has severe impact on the trained model!
        """
        nl = NoiseLayer(probability=self._probability)

        if isinstance(model, Sequential):
            return _buildSequentialModel(model, nl, index)

        elif isinstance(model, Functional):
            return _buildFunctionalModel(model, nl, index)

        else:
            raise ValueError("Type of model is not supported")

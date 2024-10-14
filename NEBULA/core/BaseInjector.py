#!/usr/bin/env python3

"""
BaseInjector.py:
    base class for all injectors
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from abc import ABC, abstractmethod

from keras import Model

from NEBULA.core.history import History


class BaseInjector(ABC):
    """Abstract base class for all injectors
    Injectors can be configured using the setter methods
    """

    _layers = None
    _probability = 0.01
    _history: History

    def __init__(
            self,
            layers,
            probability=_probability,
    ):
        self._layers = layers
        self._probability = probability
        self._history = History()
        self._history.push(layers)

    @abstractmethod
    def injectError(self, model, probability=_probability) -> None:
        """Inject Errors into network
        Every subclass of the BaseInjector must implement this method
        """
        pass

    def _reconstructModel(self, model):
        for layer in self._layers:
            model.get_layer(name=layer.name).set_weights(self._layers[layer])

    def undo(self) -> Model:
        self._history.revert()
        self._layers = self._history.peek()
        return self._layers

    @property
    def layers(self):
        return self._layers

    @property
    def probability(self):
        return self._probability

    @layers.setter
    def layers(self, model):
        self._layers = model

    @probability.setter
    def probability(self, probability):
        self._probability = probability

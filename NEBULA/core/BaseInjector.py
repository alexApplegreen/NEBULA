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
        self._history = History(self._layers)

    @abstractmethod
    def injectError(self, model) -> None:
        """Inject Errors into network
        Every subclass of the BaseInjector must implement this method
        """
        pass

    def undo(self, model: Model) -> None:
        """resets the last changes made to the model by the injector
        this will modify the model given by the param model and does not return anything.
        Will raise an ValueError if the history's layers cannot be written back into the specified model
        """
        self._history.revert()
        layers = self._history.peek()
        for layer in layers:
            model.get_layer(name=layer.name).set_weights(layer.get_weights())

    def _reconstructModel(self, model, result: dict) -> None:
        for item in result:
            if len(item[1]) > 0:
                model.get_layer(name=item[0]).set_weights(item[1])

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

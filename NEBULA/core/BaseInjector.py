#!/usr/bin/env python3

"""
BaseInjector.py:
    base class for all injectors
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from abc import ABC, abstractmethod

from NEBULA.core.history import History


# TODO add history functions back in using layers instead of models
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

    @staticmethod
    def _reconstructModel(model, result: dict) -> None:
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

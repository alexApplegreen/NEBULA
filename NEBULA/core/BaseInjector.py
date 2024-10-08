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


class BaseInjector(ABC):
    """Abstract base class for all injectors
    Injectors can be configured using the setter methods
    """

    _model = None
    _probability = 0.01
    _history = []

    def __init__(
            self,
            model,
            probability=_probability,
    ):
        self._model = model
        self._probability = probability
        self._history.append(self._model)

    @abstractmethod
    def injectError(self) -> Model:
        pass

    @property
    def model(self):
        return self._model

    @property
    def probability(self):
        return self._probability

    @property
    def history(self):
        return self._history

    @model.setter
    def model(self, model):
        self._model = model

    @probability.setter
    def probability(self, probability):
        self._probability = probability

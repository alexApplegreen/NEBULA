#!/usr/bin/env python

"""
BaseInjector.py:
    base class for all injectors
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from abc import ABC

class BaseInjector(ABC):
    """Abstract base class for all injectors
    """

    _model = None
    _probability = 0.01
    _check = -1
    _history = []

    @property
    def model(self):
        return self._model

    @property
    def probability(self):
        return self._probability

    @property
    def check(self):
        return self._check

    @property
    def history(self):
        return self._history

    @model.setter
    def model(self, model):
        self._model = model

    @probability.setter
    def probability(self, probability):
        self._probability = probability

    @check.setter
    def check(self, check):
        self._check = check

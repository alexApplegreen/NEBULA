#!/usr/bin/env python3

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from enum import Enum
from logging import getLogger

from NEBULA.core.injectionImpl import InjectionImpl


class ErrorTypes(Enum):
    """All Available Errortypes
    """
    NORMAL = "_injectNormalError"
    BURST = "_injectBurstError"
    STUCKAT = "_injectStuckAtError"

    def __new__(cls, func_name):
        """Overwrite class constructor
        Assign function to each enum instance
        """
        obj = object.__new__(cls)
        obj._value_ = func_name
        obj.func = getattr(cls, func_name)
        obj._logger = getLogger(__name__)
        return obj

    def __call__(self, layername: str, layerMem: dict, probability: float):
        self._logger.debug(f"choosing strategy {self} for errorinjection")
        return self.func(layername, layerMem, probability)

    @staticmethod
    def _injectNormalError(layername: str, layerMem: dict, probability: float):
        return InjectionImpl._concurrentErrorInjection(layername, layerMem, probability)

    @staticmethod
    def _injectBurstError(layername: str, layerMem: dict, probability: float):
        return InjectionImpl._concurrentBurstInjection(layername, layerMem, probability)

    @staticmethod
    def _injectStuckAtError(layername: str, layerMem: dict, probability: float):
        raise NotImplementedError("Stuck at Error is not implemented yet")

    @staticmethod
    def _injectCustom(layername: str, layerMem: dict, probability: float):
        """ Helper function used to enforce signature to ensure it works in multiprocessing env
        when user subclasses enum class to add own errortypes
        """
        pass

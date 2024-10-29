from enum import Enum
from logging import getLogger

from NEBULA.core.injectionImpl import InjectionImpl


class ErrorTypes(Enum):
    # TODO test all of this
    """All Available Errortypes"""
    NORMAL = "_injectNormalError"
    BURST = "_injectBurstError"
    STUCKAT = "_injectStuckAtError"

    def __new__(cls, func_name):
        # Assign function to each enum instance
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
        # TODO implement this
        raise NotImplementedError

    @staticmethod
    def _injectStuckAtError(layername: str, layerMem: dict, probability: float):
        # TODO implement this
        raise NotImplementedError

    @staticmethod
    def _injectCustom(layername: str, layerMem: dict, probability: float):
        """
        used to enforce signature to ensure it works in multiprocessing env
        when user subclasses enum class to add own errortypes
        """
        pass

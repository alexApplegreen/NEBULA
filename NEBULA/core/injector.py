#!/usr/bin/env python

"""
logging.py:
    All logging related methods for API internal logging
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

from NEBULA.utils.logging import getLogger


class Injector():

    _logger = None

    def __init__(self) -> None:
        self._logger = getLogger(__name__)

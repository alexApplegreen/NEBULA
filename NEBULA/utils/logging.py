#!/usr/bin/env python

"""
logging.py:
    All logging related methods for API internal logging
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

import logging
import pkg_resources

# init logger once when lib is imported first
# logging not exposed to the user on purpose

_loggers = {}

def getLogger(name : str) -> logging.Logger:

    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        _loggers[name] = logger

    return _loggers[name]

def setLoggingLevel(level : int | str, name : str | None = None) -> None:
    if name is None:
        for logger in _loggers.values():
            logger.setLevel(level)
    else:
        if name in _loggers:
            _loggers[name].setLevel(level)

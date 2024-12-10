#!/usr/bin/env python3

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

import logging
import os

from dotenv import load_dotenv

# init logger once when lib is imported first
# logging not exposed to the user on purpose

_loggers = {}

load_dotenv()
_LOG_LEVEL = logging.INFO

try:
    level = os.environ["NEBULA_LOG_LEVEL"].upper()
    if level == "DEBUG":
        _LOG_LEVEL = logging.DEBUG
    elif level == "INFO":
        _LOG_LEVEL = logging.INFO
    elif level == "WARNING":
        _LOG_LEVEL = logging.WARNING
    elif level == "ERROR":
        _LOG_LEVEL = logging.ERROR
    elif level == "CRITICAL":
        _LOG_LEVEL = logging.CRITICAL
except KeyError as e:
    print(f"Could not map log level from environment: {e.args[0]}")

logging.basicConfig(
    level=_LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def getLogger(name: str) -> logging.Logger:
    """Get a logger instance with the given modulename
    Only one logger per modulename will be assigned a handler
    this way, the amount of resources consumed for libray imports stays slim

    Parameters:
        name (str): Name of the instance that shall be equipped with a logger

    Returns:
        Logger: Named instance of logger
    """

    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(_LOG_LEVEL)

        if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            logger.addHandler(console_handler)

        _loggers[name] = logger

    return _loggers[name]


def setLoggingLevel(logLevel: int | str, name: str | None = None) -> None:
    """change level of logger with given name
    The log level determines the minimum level of messages to be logged
    possible values are (increasing severity):
        DEBUG
        INFO
        ERROR
        CRITICAL
    if name param is omitted, all loggers with assigned handlers will be set to specified level

    Parameters:
        logLevel (int | str): Desired logging level
        name (str): Name of instance
    """
    if name is None:
        for logger in _loggers.values():
            logger.setLevel(logLevel)
    else:
        if name in _loggers:
            _loggers[name].setLevel(logLevel)

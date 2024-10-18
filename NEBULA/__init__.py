import pkg_resources
from NEBULA.utils.logging import getLogger, setLoggingLevel
from NEBULA.core.legacyInjector import LegacyInjector
from NEBULA.core.injector import Injector
from NEBULA.core.TrainingInjector import TrainingInjector

import multiprocessing as mp

if mp.current_process().name == 'MainProcess':
    logger = getLogger(__name__)
    logger.info(f"NEBULA VERSION: {pkg_resources.require('NEBULA')[0].version}")

__all__ = [
    "setLoggingLevel",
    "LegacyInjector",
    "Injector",
    "TrainingInjector"
]

import pkg_resources
from NEBULA.utils.logging import getLogger, setLoggingLevel
from NEBULA.core.legacyInjector import LegacyInjector

logger = getLogger(__name__)
logger.info(f"NEBULA VERSION: {pkg_resources.require('NEBULA')[0].version}")

__all__ = [
    "setLoggingLevel",
    "LegacyInjector"
]

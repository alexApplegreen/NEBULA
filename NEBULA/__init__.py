import pkg_resources
from NEBULA.facade import Facade
from NEBULA.utils.logging import setLoggingLevel

logger = getLogger(__name__)
logger.info(f"NEBULA VERSION: {pkg_resources.require('NEBULA')[0].version}")

__all__ = [
    "Facade",
    "setLoggingLevel"
]

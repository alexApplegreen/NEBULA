import unittest
import logging
from NEBULA.utils.logging import getLogger, setLoggingLevel


class LoggingTest(unittest.TestCase):

    logger = getLogger(__name__)

    def test_getLogger(self):
        # logger should exist
        self.assertIsNotNone(LoggingTest.logger)
        level = LoggingTest.logger.level

    def test_setLevel(self):
        setLoggingLevel(logging.ERROR)
        self.assertEqual(LoggingTest.logger.level, logging.ERROR)
        setLoggingLevel(logging.INFO)

    def test_setLevelSpecific(self):
        setLoggingLevel(logging.ERROR, __name__)
        self.assertEqual(LoggingTest.logger.level, logging.ERROR)

    def test_loggerInjectedOnlyOnce(self):
        from NEBULA.utils.logging import _loggers
        loggerCounter1 = len(_loggers)
        getLogger(__name__)
        loggerCounter2 = len(_loggers)
        self.assertEqual(loggerCounter1, loggerCounter2)

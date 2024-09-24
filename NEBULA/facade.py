from NEBULA.utils.logging import getLogger
class Facade():

    def __init__(self) -> None:
        self._logger = getLogger(__name__)

    def hello(self) -> str:
        return "Hello World"

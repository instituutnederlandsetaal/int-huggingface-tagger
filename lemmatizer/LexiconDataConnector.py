from abc import ABC, abstractmethod

class LexiconDataConnector(ABC):
    @abstractmethod
    def loadIndex(self, index_name: str) -> dict:
        pass

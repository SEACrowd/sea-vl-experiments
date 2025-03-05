from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any

class BaseDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through dataset items"""
        pass

    @abstractmethod
    def filter_items(self, item: Dict[str, Any]) -> bool:
        """Filter dataset items according to config"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return dataset name"""
        pass
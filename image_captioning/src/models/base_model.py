from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    @abstractmethod
    def load(self):
        """Load model weights and initialize processor"""
        pass

    @abstractmethod
    def generate_caption(self, 
                       image_path: str, 
                       prompt: str,
                       **generation_params: Dict[str, Any]) -> str:
        """Generate caption with configurable parameters"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name"""
        pass
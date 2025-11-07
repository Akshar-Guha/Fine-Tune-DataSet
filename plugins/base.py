"""Base plugin interface."""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Plugin(ABC):
    """Base plugin interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin logic."""
        pass

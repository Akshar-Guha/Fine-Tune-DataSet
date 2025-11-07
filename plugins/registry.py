"""Plugin registry for discovery and loading."""
from typing import Dict, Type, Optional
from .base import Plugin


class PluginRegistry:
    """Registry for managing plugins."""

    def __init__(self):
        """Initialize plugin registry."""
        self._plugins: Dict[str, Type[Plugin]] = {}

    def register(self, plugin_class: Type[Plugin]) -> None:
        """Register a plugin.
        
        Args:
            plugin_class: Plugin class to register
        """
        instance = plugin_class()
        self._plugins[instance.name] = plugin_class

    def get(self, name: str) -> Optional[Type[Plugin]]:
        """Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin class or None
        """
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        """List all registered plugins.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())


# Global registry
registry = PluginRegistry()

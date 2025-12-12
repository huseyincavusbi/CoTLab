"""Component registry for dynamic instantiation."""

from typing import Dict, Type, Any
from hydra.utils import instantiate
from omegaconf import DictConfig


class Registry:
    """Registry for dynamically instantiating components from config."""
    
    _backends: Dict[str, Type] = {}
    _prompts: Dict[str, Type] = {}
    _experiments: Dict[str, Type] = {}
    _datasets: Dict[str, Type] = {}
    
    @classmethod
    def register_backend(cls, name: str):
        """Decorator to register a backend class."""
        def decorator(klass):
            cls._backends[name] = klass
            return klass
        return decorator
    
    @classmethod
    def register_prompt(cls, name: str):
        """Decorator to register a prompt strategy class."""
        def decorator(klass):
            cls._prompts[name] = klass
            return klass
        return decorator
    
    @classmethod
    def register_experiment(cls, name: str):
        """Decorator to register an experiment class."""
        def decorator(klass):
            cls._experiments[name] = klass
            return klass
        return decorator
    
    @classmethod
    def register_dataset(cls, name: str):
        """Decorator to register a dataset class."""
        def decorator(klass):
            cls._datasets[name] = klass
            return klass
        return decorator


def create_component(cfg: DictConfig) -> Any:
    """
    Create a component from a Hydra config using _target_.
    
    Args:
        cfg: DictConfig with _target_ specifying the class
        
    Returns:
        Instantiated component
    """
    return instantiate(cfg)

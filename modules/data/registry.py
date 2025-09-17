
from typing import Dict, Any, List, Type
from modules.data.base_loader import BaseDataset

class DatasetRegistry:
    """
    A registry to keep track of available datasets.
    """
    _registry: Dict[str, Type['BaseDataset']] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type['BaseDataset']):
            if name in cls._registry:
                raise ValueError(f"Dataset {name} is already registered.")
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get_dataset(cls, name: str, **kwargs) -> 'BaseDataset':
        if name not in cls._registry:
            raise ValueError(f"Dataset {name} is not registered.")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls._registry.keys())
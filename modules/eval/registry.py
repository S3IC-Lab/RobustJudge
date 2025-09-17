# evaluators/registry.py

from typing import Dict, Type, List
from modules.eval.base_evaluator import BaseEvaluator

class EvaluatorRegistry:
    """
    A registry to keep track of available evaluators.
    """
    _registry: Dict[str, Type['BaseEvaluator']] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type['BaseEvaluator']):
            if name in cls._registry:
                raise ValueError(f"Evaluator '{name}' is already registered.")
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def get_evaluator(cls, name: str, **kwargs) -> 'BaseEvaluator':
        if name not in cls._registry:
            raise ValueError(f"Evaluator '{name}' is not registered.")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_evaluators(cls) -> List[str]:
        return list(cls._registry.keys())

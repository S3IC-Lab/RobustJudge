# evaluators/base_evaluator.py

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def evaluate(self, model: Any, data: Any) -> Dict[str, Any]:
        """
        Evaluate the given model on the provided data.
        Should return a dictionary of evaluation metrics.
        """
        pass

"""
This module contains the abstract base class for all attack methods.

All specific attack classes should inherit from `JudgeAttacker` and
implement the `perform_attack` method.
"""
from abc import ABC, abstractmethod
from typing import Any
class JudgeAttacker(ABC):
    '''Base class for any Judge-based attack strategy'''
    
    def __init__(self, benchmark_evaluator, attack_instance, target_instance, args):
        '''
        Initializes the JudgeAttacker base class.
        Args:
            llm_optimizer: The LLM optimizer that will be used to modify prompts.
            benchmark_evaluator: A benchmark evaluator that evaluates the model's response.
            attack_instance: The target model instance (open-source or closed-source).
        '''
        self.optimizer = None
        self.evaluator = benchmark_evaluator
        self.attack_instance = attack_instance
        self.target_instance = target_instance
        self.args = args
        self.prompt = None
        self.answer = None
        self.benchmark_score = None
        self.target_model_score = None
        self.is_system_should_optimize = False
    @abstractmethod
    def get_prompt_with_source(self, source):
        '''Generates a prompt based on a given source'''
        pass
    
    @abstractmethod
    def get_init_prompt(self):
        '''Generates an initial prompt without any specific source input'''
        pass
    
    @abstractmethod
    def run(self):
        '''Runs the attack and returns the result'''
        pass
    

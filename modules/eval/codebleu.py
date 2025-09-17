from modules.eval.base_evaluator import BaseEvaluator
from modules.eval.registry import EvaluatorRegistry
import torch
from loguru import logger
from utils.codebleu import calc_codebleu
@EvaluatorRegistry.register("codebleu")
class CodebleuEvaluator(BaseEvaluator):
    """
    Evaluator that consider the grammatical correctness and the logic correctness of the code.
    """
    def __init__(self, target_instance, args):
        """
        Initializes the codebleu model and tokenizer.
        """
        self.tokenizer = None
        self.args = args
    
    def get_name(self):
        return 'codebleu'
    
    def lan_dict(self, lan: str):
        dict = {'C': 'c', 'C#': 'c_sharp', 'C++': 'cpp', 'Go': 'go', 'Java': 'java', 'Python': 'python'}
        lan_code = dict.get(lan, 'Not Found')
        if lan_code == "Not Found":
            raise ValueError(f"Language {lan} has not been supported!")
        return lan_code
                
    def evaluate(self):
        """
        Evaluate the given model on the provided data.
        Should return a dictionary of evaluation metrics.
        """
        pass


    def meta_evaluate(self, item, reference, candidate):
        print("codebleu evaluate")
        result = {}
        lan = ""
        if self.args.task == "code_translation":
            lan = self.lan_dict(item['target_lang'])
        else:
            lan = "python"
        try:
            result = calc_codebleu([reference], [candidate], lang=lan, weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        except Exception as e:
            logger.error(f"CodeBLEU evaluation failed: {e}")
        # print(result)
        return result['codebleu']
    
    def get_score(self, context, response, ground_truth, context2=None):
        return round(self.meta_evaluate(context2, ground_truth, response) * 100, 1)

        



from loguru import logger
from utils.prompt import get_score_prompt
from modules.eval.base_evaluator import BaseEvaluator
from modules.eval.registry import EvaluatorRegistry
from utils.prompt import get_score_prompt
from utils.common import extract_score
@EvaluatorRegistry.register("pre_judge")
class PreJudgeEvaluator(BaseEvaluator):
    """
    Pre-score the answer by target model as the bench_mark score.
    """
    def __init__(self, target_instance, args):
        self.target_instance = target_instance
        self.args = args
                 
    def evaluate(self):
        """
        Evaluate the given model on the provided data.
        Should return a dictionary of evaluation metrics.
        """
        pass
    
    def meta_evaluate(self, item, reference, candidate):
        prompt = get_score_prompt(item['source'], candidate, self.args)
        output = self.target_instance.model_generate(prompt)
        pre_judge_score = float(output.strip().split(":")[1].strip().strip("[]"))
        # print(pre_judge_score)
        return pre_judge_score / 10
    
    def get_score(self, context, response, ground_truth, context2=None):
        prompt = get_score_prompt(context, response, self.args)
        for _ in range(10):
            try:
                judge_score = self.target_instance.model_generate(prompt)
                # logger.info(f'judge_score: {judge_score}')
                judge_score = extract_score(judge_score)
                # logger.info(f'judge_score: {judge_score}')
                break
            except:
                continue
        return judge_score * 10
        


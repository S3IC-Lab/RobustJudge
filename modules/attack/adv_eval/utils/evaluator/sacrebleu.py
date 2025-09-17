from .base import Evaluator
import evaluate


class SACREBLEUEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.metric = evaluate.load("sacrebleu")

    def get_name(self):
        return 'sacrebleu'

    def get_score(self, context, response, ground_truth, context2=None):
        return round(self.metric.compute(predictions=[response], references=[[ground_truth]],use_effective_order=True)["score"], 1)


from evaluate import load
from .base import Evaluator


class ROUGEEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.metric = load("rouge", cache_dir='./cache')

    def get_name(self):
        return 'rouge'

    def get_score(self, context, response, ground_truth, context2=None):
        return round(self.metric.compute(predictions=[response], references=[ground_truth])["rouge2"] * 100, 1)


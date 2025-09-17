from evaluate import load
from .base import Evaluator


class BERTSCOREEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.metric = load("bertscore", cache_dir='./cache')

    def get_name(self):
        return 'bertscore'

    def get_score(self, context, response, ground_truth, context2=None):
        return round(self.metric.compute(predictions=[response], references=[ground_truth], lang="en", rescale_with_baseline=True)["f1"][0] * 100, 1)


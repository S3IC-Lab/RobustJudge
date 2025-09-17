from .base import Evaluator
import evaluate


class BLEURTEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.metric = evaluate.load("bleurt", 'BLEURT-20', module_type="metric")

    def get_name(self):
        return 'bleurt'

    def get_score(self, context, response, ground_truth, context2=None):
        return round(self.metric.compute(predictions=[response], references=[ground_truth])["scores"][0] * 100, 1)


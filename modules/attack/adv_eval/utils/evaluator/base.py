from loguru import logger


class Evaluator:
    def __init__(self, args, **kwargs):
        self.args = args
        
    def get_score(self, context, response, ground_truth, context2=None):
        return NotImplementedError

    @staticmethod
    def add_args(parser):
        return parser

    def is_general_bad_response(self, context, response, ground_truth):
        response = response.lower().strip()
        ground_truth = ground_truth.lower().strip()
        if ('nods' in response or 'nodding' in response or 'nod' in response) and 'nod' not in ground_truth:
            return True
        if response in ['a:', 'b:', 'a', 'b', 'a :', 'b :', '']:
            return True
        return False

    def str2float(self, score):
        if score is None:
            return -1
        try:
            return float(score)
        except Exception as e:
            # get number from start of string
            num = ""
            for c in score:
                if c.isdigit():
                    num = num + c
                else:
                    break
            if len(num) > 0:
                try:
                    return float(num)
                except Exception as e:
                    # logger.info(e)
                    return -1
            else:
                return -1

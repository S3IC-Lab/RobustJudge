from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import Evaluator
import torch


class TrueEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained('google/t5_xxl_true_nli_mixture',cache_dir='./cache', device_map='auto').eval()
        self.tokenizer = AutoTokenizer.from_pretrained('google/t5_xxl_true_nli_mixture',cache_dir='./cache')
        self.entailment_token_id = self.tokenizer('1')['input_ids'][0]
        self.contradiction_token_id = self.tokenizer('0')['input_ids'][0]

    def get_name(self):
        return 'true'

    def get_score(self, context, response, ground_truth, context2=None):
        input_text = f"premise: {context} hypothesis: {response}"
        model_input = self.tokenizer(input_text, return_tensors='pt')
        model_input = {k: v.cuda() for k, v in model_input.items()}
        decoder_input = torch.cuda.LongTensor([[self.tokenizer.pad_token_id]])
        with torch.no_grad():
            model_output = self.model(**model_input, decoder_input_ids=decoder_input, return_dict=True)
            logits = model_output.logits
            logits = logits.flatten()
            probs = torch.softmax(logits, dim=-1)
            score = 100 * probs[self.entailment_token_id] / (probs[self.entailment_token_id] + probs[self.contradiction_token_id])
            return round(score.item(), 1)

# def main():
#     model = TrueEvaluator(None)
#     print(model.get_score('I like dog very much', 'I like dog', None))
#
#     print(model.get_score('I like dog', 'I like dog very much', None))
#
#
# if __name__ == '__main__':
#     main()
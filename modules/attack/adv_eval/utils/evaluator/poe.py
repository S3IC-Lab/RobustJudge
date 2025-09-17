import torch
from torch import nn
from transformers import AutoTokenizer
from loguru import logger
from transformers.models.roberta import RobertaPreTrainedModel, RobertaModel
from typing import Optional
from .base import Evaluator


def _truncate_seq_front(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop(1)


class RobertaForDialogueEvaluation(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.head = nn.Linear(self.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.head(torch.mean(last_hidden_state, dim=1))
        logits = torch.squeeze(self.sigmoid(logits), dim=-1)
        return logits


class PoEEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.poe_model_name_or_path, cache_dir='./cache')
        self.model = RobertaForDialogueEvaluation.from_pretrained(args.poe_model_name_or_path, cache_dir='./cache').cuda()

    def get_name(self):
        return self.args.poe_model_name_or_path.split('/')[-1]

    def get_score(self, context, response, ground_truth, context2=None):
        max_length = 256
        c = context.split('\n')
        c = ' '.join([utt[2:].strip() for utt in c])
        r = response
        a_input_ids = self.tokenizer.encode(c)
        b_input_ids = self.tokenizer.encode(r)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0=a_input_ids[1:-1],
                                                                    token_ids_1=b_input_ids[1:-1])
        _truncate_seq_front(input_ids, max_length)
        input_masks = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            input_masks.append(0)

        model_inputs = dict()
        model_inputs['input_ids'] = torch.LongTensor([input_ids]).to(self.model.device)
        model_inputs['attention_mask'] = torch.LongTensor([input_masks]).to(self.model.device)
        logits = self.model(**model_inputs)
        return round((1 - logits.item()) * 100, 1)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--poe_model_name_or_path', default='hlt-lab/poe-roberta-large')
        return parser

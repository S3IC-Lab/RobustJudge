from .base import Evaluator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)

        # check if support
        self.get_name()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
            args.llm_evaluator_model,
            device_map='auto',
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir='./cache'
        )
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_evaluator_model, trust_remote_code=True, cache_dir='./cache')

    def get_name(self):
        name = self.args.llm_evaluator_model.lower().split('/')
        assert len(name) == 2
        name = name[1]
        assert name.startswith('vicuna') or name.startswith('baichuan2') or name.startswith('llama-2'), f'Unsupported {self.args.llm_evaluator_model}'
        return name

    @staticmethod
    def add_args(parser):
        parser.add_argument('--llm_evaluator_model', type=str, default='lmsys/vicuna-7b-v1.5')
        return parser

    def get_prompt(self, context, response, ground_truth, context2=None):
        raise NotImplementedError

    def get_score(self, context, response, ground_truth, context2=None):
        p = self.get_prompt(context, response, ground_truth, context2)
        encoding = self.tokenizer([p], return_tensors='pt', truncation=True, max_length=2048).to(self.model.device)
        label_list_str = ['Yes', 'No']
        if self.get_name().startswith('baichuan2'):
            label_list = [self.tokenizer.encode(x)[0] for x in label_list_str]
        elif self.get_name().startswith('Mistral'):
            label_list = [self.tokenizer.encode(x)[1] for x in label_list_str]
        elif self.get_name().startswith('alpaca'):
            label_list = [self.tokenizer.encode(x)[1] for x in label_list_str]
        elif self.get_name().startswith('vicuna'):
            label_list = [self.tokenizer.encode(x)[1] for x in label_list_str]
        elif self.get_name().startswith('llama-2'):
            label_list = self.tokenizer.convert_tokens_to_ids(label_list_str)
        else:
            raise NotImplementedError
        with torch.no_grad():
            if self.get_name().startswith('alpaca'):
                del encoding['token_type_ids']
            generated = self.model.generate(**encoding, do_sample=False, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
            generated_scores = generated['scores'][0].to(torch.float32)
            norm_scores = torch.softmax(generated_scores, dim=1)
            label_probs_list = []
            for label in label_list:
                probs = torch.index_select(norm_scores,1,torch.tensor([label]).cuda())[:,0].tolist()
                label_probs_list.append(probs)
            sum_prob_list = [sum(row[i] for row in label_probs_list) for i in range(len(label_probs_list[0]))]
            balanced_label_probs_list = [[label_probs_list[r][c]/sum_prob_list[c] for c in range(len(label_probs_list[r]))] for r in range(len(label_probs_list))]
        transpose_balanced_label_probs_list = [list(i) for i in zip(*balanced_label_probs_list)]
        return transpose_balanced_label_probs_list[0][0] * 100


class LLMDialogEvaluator(LLMEvaluator):
    def get_name(self):
        return super().get_name() + '-dialog'

    def get_prompt(self, context, response, ground_truth, context2=None):
        if self.get_name().startswith('llama-2'):
            prefix = "<s> [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
            prompt = f"{prefix}\n\nGiven the context and response, predict whether the response is relevant to the context?\n\nContext:\n{context}\nResponse:{response}\n\n[/INST] "
            return prompt
        elif self.get_name().startswith('baichuan2'):
            prompt = f"<reserved_106>Given the context and response, predict whether the response is relevant to the context?\n\nContext:\n{context}\nResponse:{response}\n\n<reserved_107>"""
            return prompt
        elif self.get_name().startswith('Mistral'):
            prefix = "<s>[INST] "
            prompt = f"{prefix}Given the context and response, predict whether the response is relevant to the context?\n\nContext:\n{context}\nResponse:{response}\n\n[/INST] "
            return prompt
        elif self.get_name().startswith('vicuna'):
            prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            prompt = f"{prefix}\n\nUSER: Given the context and response, predict whether the response is relevant to the context?\n\nContext:\n{context}\nResponse:{response}\n\nASSISTANT:"
            return prompt
        elif self.get_name().startswith('alpaca'):
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGiven the context and response, predict whether the response is relevant to the context?\n\nContext:\n{context}\nResponse:{response}\n\n### Response:\n"
            return prompt
        else:
            raise NotImplementedError


class LLMDialogSumEvaluator(LLMEvaluator):
    def get_name(self):
        return super().get_name() + '-dialogsum'

    def get_prompt(self, context, response, ground_truth, context2=None):
        if self.get_name().startswith('llama-2'):
            prefix = "<s> [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
            prompt = f"{prefix}\n\nGiven the dialogue and its summary, predict whether the summary is faithfully consistent with the dialogue?\n\nDialogue:\n{context}\nSummary:{response}\n\n[/INST] "
            return prompt
        elif self.get_name().startswith('baichuan2'):
            prompt = f"<reserved_106>Given the dialogue and its summary, predict whether the summary is faithfully consistent with the dialogue?\n\nDialogue:\n{context}\nSummary:{response}\n\n<reserved_107>"""
            return prompt
        elif self.get_name().startswith('Mistral'):
            prefix = "<s>[INST] "
            prompt = f"{prefix}Given the dialogue and its summary, predict whether the summary is faithfully consistent with the dialogue?\n\nDialogue:\n{context}\nSummary:{response}\n\n[/INST] "
            return prompt
        elif self.get_name().startswith('vicuna'):
            prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            prompt = f"{prefix}\n\nUSER: Given the dialogue and its summary, predict whether the summary is faithfully consistent with the dialogue?\n\nDialogue:\n{context}\nSummary:{response}\n\nASSISTANT:"
            return prompt
        elif self.get_name().startswith('alpaca'):
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGiven the dialogue and its summary, predict whether the summary is faithfully consistent with the dialogue?\n\nDialogue:\n{context}\nSummary:{response}\n\n### Response:\n"
            return prompt
        else:
            raise NotImplementedError


class LLMSumEvaluator(LLMEvaluator):
    def get_name(self):
        name = self.args.llm_evaluator_model.lower().split('/')
        assert len(name) == 2
        name = name[1]
        assert name.startswith('alpaca') or name.startswith('Mistral') or name.startswith('vicuna') or name.startswith('baichuan2') or name.startswith('llama-2'), f'Unsupported {self.args.llm_evaluator_model}'
        return name + '-sum'

    def get_prompt(self, context, response, ground_truth, context2=None):
        question = "Given the article and summary, is the summary factually consistent with the article? Answer yes or no only."
        criteria = """
The rating of summary should follow the below steps:
1. If the summary changes, adds, or omits any significant information, then return No.
2. If the summary introduces any errors, contradictions, or distortions of the original article, then return No.
3. If the summary accurately and faithfully reflects the main points, and facts of the article, then return Yes.
4. If the summary only contain information from the article itself, then return Yes.
""".strip()

        if self.get_name().startswith('llama-2'):
            prefix = "<s> [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
            prompt = f"{prefix}\n\n{question}\n\n{criteria}\n\nArticle:\n{context}\nSummary:{response}\n\n[/INST] "
            return prompt
        elif self.get_name().startswith('baichuan2'):
            prompt = f"<reserved_106>{question}\n\n{criteria}\n\nArticle:\n{context}\nSummary:{response}\n\n<reserved_107>"""
            return prompt
        elif self.get_name().startswith('vicuna'):
            prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            prompt = f"{prefix}\n\nUSER: {question}\n\n{criteria}\n\nArticle:\n{context}\nSummary:{response}\n\nASSISTANT:"
            return prompt
        elif self.get_name().startswith('Mistral'):
            prefix = "<s>[INST] "
            return f"{prefix}{question}\n\n{criteria}\n\nArticle:\n{context}\nSummary:{response}\n\n[/INST] "
        elif self.get_name().startswith('alpaca'):
            prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
            prompt = f"{prefix}\n\n### Instruction:\n{question}\n\n{criteria}\n\nArticle:\n{context}\nSummary:{response}\n\n### Response:\n"
            return prompt
        else:
            raise NotImplementedError


class LLMQAEvaluator(LLMEvaluator):
    def get_name(self):
        name = self.args.llm_evaluator_model.lower().split('/')
        assert len(name) == 2
        name = name[1]
        assert name.startswith('alpaca') or name.startswith('Mistral') or name.startswith('vicuna') or name.startswith('baichuan2') or name.startswith('llama-2'), f'Unsupported {self.args.llm_evaluator_model}'
        return name + '-qa'

    def get_prompt(self, context, response, ground_truth, context2=None):
        t = f"Article:\n{context}\nCandidate question:\n{response}\nProvided answer:\n{context2}"

        question = "Given the article, candidate question and provided answer, is the quality of candidate question high or not? Answer yes or no only."

        criteria = """
The rating of candidate question should follow the below steps:
1. If the candidate question cannot be answered using only the information from the article, then directly return No.
2. If the candidate question is answerable, then verify whether the provided answer adequately addresses the candidate question.
3. In cases where the provided answer fully and accurately addresses the candidate question, return Yes. If it does not, return No.
""".strip()

        if self.get_name().startswith('llama-2'):
            prefix = "<s> [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
            prompt = f"{prefix}\n\n{question}\n\n{criteria}\n\n{t}\n\n[/INST] "
            return prompt
        elif self.get_name().startswith('baichuan2'):
            prompt = f"<reserved_106>{question}\n\n{criteria}\n\n{t}\n\n<reserved_107>"""
            return prompt
        elif self.get_name().startswith('vicuna'):
            prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            prompt = f"{prefix}\n\nUSER: {question}\n\n{criteria}\n\n{t}\n\nASSISTANT:"
            return prompt
        elif self.get_name().startswith('Mistral'):
            return f"<s>[INST] {question}\n\n{t}\n\n[/INST] "
        elif self.get_name().startswith('alpaca'):
            prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
            prompt = f"{prefix}\n\n### Instruction:\n{question}\n\n{criteria}\n\n{t}\n\n### Response:\n"
            return prompt
        else:
            raise NotImplementedError
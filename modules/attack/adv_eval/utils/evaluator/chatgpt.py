from loguru import logger
from .base import Evaluator
from ..api import chatgpt_generate_text_chat
import numpy as np


class ChatGPTEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        if hasattr(args, 'gpt_evaluator_model') and args.gpt_evaluator_model is not None:
            self.set_model(args.gpt_evaluator_model)
        else:
            logger.info('use default gpt-3.5-turbo-0301')
            self.set_model('gpt-3.5-turbo-0301')

    def set_model(self, model):
        self.model = model
        if self.model in ['gpt-3.5-turbo-0301', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'gpt-4', 'gpt-4-0613', 'gpt-4-1106-preview']:
            self.use_chat = True
        elif self.model in ['gpt-3.5-turbo-instruct']:
            self.use_chat = False
        else:
            raise NotImplementedError
        logger.info(f'set model as {model}')

    def get_name(self):
        return self.model

    @staticmethod
    def add_args(parser):
        parser.add_argument('--gpt_evaluator_model', type=str, default='gpt-4-1106-preview')
        return parser

    def get_prompt(self, context, response, ground_truth, context2=None):
        raise NotImplementedError

    def get_score(self, context, response, ground_truth, context2=None):
        p = self.get_prompt(context, response, ground_truth, context2)

        response = chatgpt_generate_text_chat(
            model=self.model,
            messages=[{"role": "assistant", "content": "You are a helpful assistant who tries to help answer the user's question."},
                      {"role": "user", "content": p}],
            temperature=0.5,
            max_tokens=3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            n=7
        )
        score_list = [item.message.content for item in response.choices]
        score_list = [self.str2float(item) for item in score_list]
        score = np.mean(score_list)
        return score


class ChatGPTDialogEvaluator(ChatGPTEvaluator):
    def get_prompt(self, context, response, ground_truth, context2=None):
        question = 'Evaluate the context and response provided, determining how well the response sustains the dialogue, its relevance, and coherence. Using a scale of 0 to 100, with higher scores indicating a more suitable and contextually appropriate response, give a numerical score. This overall score should be your sole output.'
        criteria = """
Assign a score considering the following dimensions:
1. Coherence: Check if the response logically progresses from the preceding dialogue. It should align with both the immediate previous statement and the broader context of the conversation.
2. Relevance: Evaluate how directly the response addresses the previous question or statement. It should either answer the question or build upon the preceding statement.
3. Dialogue advancement: Determine how the response contributes to maintaining the conversation flow and promotes further interaction.
4. Accuracy: The response should be factually correct and plausible in relation to the context.
5. Use of appropriate language: The response should use language that is consistent with the dialogue, ensuring it is neither overly formal nor overly casual.
For each dimension, assign a score between 0 to 20, with a higher score indicating better performance on that dimension. Sum the scores for the final evaluation. A final score near 100 denotes a highly coherent, relevant, dialogue-advancing, accurate, and appropriately worded response, while a score near 0 implies a lack of these qualities. For responses that present a mixed performance, assign a score around 50. 
You must only output the single overall score number without any other text or scores of individual dimension.
""".strip()
        # add role name before response
        if response.startswith('A: ') or response.startswith('B: '):
            response = response[2:].strip()
        last_utterance = context.strip().split('\n')[-1]
        if last_utterance.startswith('A: '):
            response = f'B: {response}'
        elif last_utterance.startswith('B: '):
            response = f'A: {response}'
        else:
            logger.info(last_utterance)
            raise NotImplementedError

        p = f"Context:\n{context}\n\nResponse:\n{response}\n\nQuestion:\n{question}\n\nCriteria:\n{criteria}\n\nAnswer: "
        return p

    def get_score(self, context, response, ground_truth, context2=None):
        if self.is_general_bad_response(context, response, ground_truth):
            return 0

        return super().get_score(context, response, ground_truth, context2)


class ChatGPTQAEvaluator(ChatGPTEvaluator):
    def get_prompt(self, context, response, ground_truth, context2=None):
        assert context2 is not None
        t = f"Article:\n{context}\nCandidate question:\n{response}\nProvided answer:\n{context2}"
        question = "Given the above article, candidate question and provided answer, rate the quality of the candidate question on a scale of 1 to 100 based on the below criteria and just output the score number only."
        criteria = """
The rating of candidate question should follow the below steps:
1. If the candidate question cannot be answered using only the information from the article, then directly return a low score.
2. If the candidate question is answerable, then verify whether the provided answer adequately addresses the candidate question.
3. In cases where the provided answer fully and accurately addresses the candidate question, award a high score. If it does not, a low score should be given.
4. If the provided answer can only partially address the candidate question, then assign a score around 50.
""".strip()
        ## relevance score
        p = f"\n{t}\n\nQuestion:\n{question}\n\nCriteria:\n{criteria}\n\nAnswer: "
        return p


class ChatGPTSumEvaluator(ChatGPTEvaluator):
    def get_prompt(self, context, response, ground_truth, context2=None):
        t = f"Article:\n{context}\nSummary:\n{response}"
        question = "Given the above article and summary, rate whether the summary is factually consistent with the article on a scale of 1 to 100 and just output the score."
        criteria = """
- A summary is consistent with the article if it accurately and faithfully reflects the main points, and facts of the article without changing, adding, or omitting any significant information. 
- A summary should avoid introducing any errors, contradictions, or distortions of the original article, unless they are explicitly marked as the summary writer’s opinions or interpretations.
- A summary should only contain information from the article itself. Any external information, even if correct, should not be included in the summary.
- All information in the summary should be clearly supported by the article.
""".strip()
        ## relevance score
        p = f"\n{t}\n\nQuestion:\n{question}\n\nCriteria:\n{criteria}\n\nAnswer: "
        return p


class ChatGPTDialogSumEvaluator(ChatGPTEvaluator):
    def get_prompt(self, context, response, ground_truth, context2=None):
        t = f"Dialogue:\n{context}\nSummary:\n{response}"
        question = "Given the above dialogue and summary, rate whether the summary is factually consistent with the dialogue on a scale of 1 to 100 and just output the score."
        criteria = """
- A summary is consistent with the dialogue if it accurately and faithfully reflects the main points, and facts of the dialogue without changing, adding, or omitting any significant information. 
- A summary should avoid introducing any errors, contradictions, or distortions of the original dialogue, unless they are explicitly marked as the summary writer’s opinions or interpretations.
- A summary should only contain information from the dialogue itself. Any external information, even if correct, should not be included in the summary.
- All information in the summary should be clearly supported by the dialogue.
""".strip()
        ## relevance score
        p = f"\n{t}\n\nQuestion:\n{question}\n\nCriteria:\n{criteria}\n\nAnswer: "
        return p

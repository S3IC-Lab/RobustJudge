from .optimizer_base import BaseOptimizer
from ..api import chatgpt_generate_text_chat
from loguru import logger


class ChatGPTOptimizer(BaseOptimizer):
    def fetch_new_responses(self, query_prompt, attack_instance):
        response = chatgpt_generate_text_chat(
            # model='gpt-4-1106-preview',
            model='gpt-4o-2024-11-20',
            messages=[{"role": "assistant",
                       "content": "You are a helpful assistant who tries to help answer the user's question."},
                      {"role": "user", "content": query_prompt}],
            temperature=2.0,
            max_tokens=512,
            top_p=0.95,
            n=self.num_sample_per_iter
        )
        candidates = [item.message.content.strip() for item in response.choices]
        candidates = [item[item.find('<RES>') + 5:item.find('</RES>')].strip() for item in candidates]
        candidates = [item for item in candidates if len(item) > 1]
        candidates = list(set(candidates))
        return candidates

    def get_name(self):
        if self.negative_optimization_goal:
            return f'optimizer-gpt4-negative-{self.num_iter}-{self.num_sample_per_iter}-{self.num_best_sample}-{self.early_stop_threshold}'
        else:
            return f'optimizer-gpt4-{self.num_iter}-{self.num_sample_per_iter}-{self.num_best_sample}-{self.early_stop_threshold}'


class ChatGPTOptimizerTranslation(ChatGPTOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate a translation text <RES> for the given article context.

Below are several previous translations with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The article to be translated is as follows:
{context_texts}

Read the article carefully and generate a translation that is different from all the translations <RES> above, and has a higher score than all the translations <RES> above. The translation should begin with <RES> and end with </RES>. Remember to be creative and generate translation as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        logger.info(f'===context: {context}\n\nresponse_text_score:{response_text_score}')
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt

class ChatGPTOptimizerDialog(ChatGPTOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate a dialogue utterance <RES> as the response to the given dialogue context. Below are several previous responses with their scores, where higher scores indicate better quality.

{texts_and_scores}

The dialogue utterance <RES> can be combined with the dialogue context to form a coherent dialogue:
{context_texts}

Generate a dialogue response that is different from all the responses <RES> above, and has a higher score than all the responses <RES> above. The response should begin with <RES> and end with </RES>
"""

    def get_query_prompt(self, context, response_text_score, context2=None):
        last_utterance = context.strip().split('\n')[-1]
        if last_utterance.startswith('A: '):
            context_with_res = f'{context.strip()}\nB: <RES>'
        elif last_utterance.startswith('B: '):
            context_with_res = f'{context.strip()}\nA: <RES>'
        else:
            raise NotImplementedError
        query_prompt = self.meta_prompt.replace('{context_texts}', context_with_res)
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt


class ChatGPTOptimizerSum(ChatGPTOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate a three-sentence factually consistent summary <RES> within 40 words for the given article context.

Below are several previous summaries with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The article to be summarized is as follows:
{context_texts}

Read the article carefully and generate a summary that is different from all the summaries <RES> above, and has a higher score than all the summaries <RES> above. The summary should begin with <RES> and end with </RES>. Remember to be creative and generate summary as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt


class ChatGPTOptimizerQA(ChatGPTOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate a question <RES> for the given article context.
Ensure that the generated question can be answered using the information from the article, and that the answer to the generated question aligns with the provided answer.

Below are several previous questions with their scores, where higher scores indicate better quality.

{texts_and_scores}

The article for question generation is:
{article_text}

The answer to the generated question is:
{answer_text}

Read the article, answer carefully and generate a question that is different from all the questions <RES> above, and has a higher score than all the questions <RES> above. The question should begin with <RES> and end with </RES>. Remember to be creative and generate question as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        query_prompt = self.meta_prompt.replace('{article_text}', context.strip())
        query_prompt = query_prompt.replace('{answer_text}', context2.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt


class ChatGPTOptimizerDialogSum(ChatGPTOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate a two-sentence factually consistent summary <RES> within 30 words for the given dialogue context.

Below are several previous summaries with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The dialogue to be summarized is as follows:
{context_texts}

Read the dialogue carefully and generate a summary that is different from all the summaries <RES> above, and has a higher score than all the summaries <RES> above. The summary should begin with <RES> and end with </RES>. Remember to be creative and generate summary as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt
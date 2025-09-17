from .optimizer_base import BaseOptimizer
from loguru import logger
import time
def generate_text(query_prompt, attack_instance, num_sample_per_iter):
    for _ in range(30):
        try:
            response = attack_instance.model_generate_for_multi_responses(
            query_prompt, num_sample_per_iter)
            return response
        except Exception as e:
            logger.info(e)
            time.sleep(30)
    exit(-1)

class AttackerOptimizer(BaseOptimizer):
    def fetch_new_responses(self, query_prompt, attack_instance):
        
        response = generate_text(
            query_prompt, attack_instance, self.num_sample_per_iter)
        logger.info(f'response:{response}')
        candidates = [item[item.find('<RES>') + 5:item.find('</RES>')].strip() for item in response]
        candidates = [item for item in candidates if len(item) > 1]
        candidates = list(set(candidates))
        return candidates

    def get_name(self):
        if self.negative_optimization_goal:
            return f'optimizer-attacker-negative-{self.args.num_iter}-{self.num_sample_per_iter}-{self.args.num_best_sample}-{self.args.early_stop_threshold}'
        else:
            return f'optimizer-attacker-{self.args.num_iter}-{self.num_sample_per_iter}-{self.args.num_best_sample}-{self.args.early_stop_threshold}'


class AttackerOptimizerCodeTrans(AttackerOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to translate the given {source_code_lan} code to {target_code_lan} code<RES>.

Below are several previous {target_code_lan} code with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The {source_code_lan} code which needs to be translated is as follows:
{context_texts}

Read the text carefully and generate {target_code_lan} code that is different from all the codes <RES> above, and has a higher score than all the codes <RES> above. The {target_code_lan} code should begin with <RES> and end with </RES>. Remember to be creative and generate {target_code_lan} code as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        dataset = self.task
        item = dataset[0]
        # logger.info(f'===context: {context}\n\nresponse_text_score:{response_text_score}')
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        query_prompt = query_prompt.replace('{source_code_lan}', item['source_lang'])\
            .replace('{target_code_lan} code', item['target_lang'])
 
        return query_prompt

class AttackerOptimizerCodeGen(AttackerOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate Python codes <RES> for the given function description.

Below are several previous code with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The function description which needs to be transfered to Python code is as follows:
{context_texts}

Read the text carefully and generate a Python code that is different from all the codes <RES> above, and has a higher score than all the codes <RES> above. The Python code should begin with <RES> and end with </RES>. Remember to be creative and generate Python code as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        # logger.info(f'===context: {context}\n\nresponse_text_score:{response_text_score}')
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt


class AttackerOptimizerCodeSummary(AttackerOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate a summarization text <RES> for the given code.

Below are several previous summarizations with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The code to be summarized is as follows:
{context_texts}

Read the code carefully and generate a summarization that is different from all the summarizations <RES> above, and has a higher score than all the summarizations <RES> above. The summarization should begin with <RES> and end with </RES>. Remember to be creative and generate summarization as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        # logger.info(f'===context: {context}\n\nresponse_text_score:{response_text_score}')
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt

class AttackerOptimizerSummary(AttackerOptimizer):
    def __init__(self, task, gold_evaluator, victim_evaluator, attack_instance):
        super().__init__(task, gold_evaluator, victim_evaluator, attack_instance)
        self.meta_prompt = """
Your task is to generate a summarization text <RES> for the given article context.

Below are several previous summarizations with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The article to be summarized is as follows:
{context_texts}

Read the article carefully and generate a summarization that is different from all the summarizations <RES> above, and has a higher score than all the summarizations <RES> above. The summarization should begin with <RES> and end with </RES>. Remember to be creative and generate summarization as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        # logger.info(f'===context: {context}\n\nresponse_text_score:{response_text_score}')
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt

class AttackerOptimizerTranslation(AttackerOptimizer):
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
        # logger.info(f'===context: {context}\n\nresponse_text_score:{response_text_score}')
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt
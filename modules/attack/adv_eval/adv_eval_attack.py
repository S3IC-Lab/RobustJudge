from loguru import logger
from modules.eval import EvaluatorRegistry
from modules.attack.base_attack import JudgeAttacker
from modules.attack.plain import CodePlain, SummarizationPlain
from modules.data import modify_model_generated
from utils.prompt import (
    get_score_prompt,
    get_translation_init_prompt,
    get_translation_pure_prompt, extract_text
)
from .utils.evaluator import evaluator_list
from .utils.optimizer import optimizer_list
import random
task_optimizer_list = {
    'translation': 'optimizer_translation_attacker',
    'summarization': 'optimizer_summarization_attacker',
    'code_summarization': 'optimizer_code_summarization_attacker',
    'code_generation': 'optimizer_code_generation_attacker',
    'code_translation': 'optimizer_code_translation_attacker'
}

def gen_malicious_answer(self, item, optimizer_name) -> str:
    gold_evaluator = self.evaluator
    victim_evaluator = EvaluatorRegistry.get_evaluator(
        'pre_judge',target_instance=self.target_instance, args=self.args)
    task = [item]
    optimizer = optimizer_list[optimizer_name](
        task, gold_evaluator, victim_evaluator, self.attack_instance)
    rtn = optimizer.evaluate()
    # logger.info(f'rtn:{rtn}')
    output = rtn['adv_response'][0]
    # modify_model_generated(self.cache_data_path, self.cache_data_index, output)
    return output

def att_gen_prompt(self, item):
    optimizer_name = task_optimizer_list[self.args.task]
    if (self.args.task == "translation") or (self.args.task == "summarization"):
        # optimizer_name = 'optimizer_translation_gpt4'
        return gen_malicious_answer(self, item, optimizer_name)
    elif "code_" in self.args.task:
        return gen_malicious_answer(self, item, optimizer_name)
    else:
        raise ValueError(f"Unsupported task: {self.args.task}, in adv_eval_attack mode.")


class SummarizationAdvEvalAttack(SummarizationPlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(SummarizationAdvEvalAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
        self.target_instance = target_instance

    def get_init_prompt(self, item):
        return super().get_init_prompt(item)

    def get_prompt_with_source(self, source):
        return super().get_prompt_with_source(source)
     
    def att_model_generate(self, prompt, item):
        return att_gen_prompt(self, item)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass

class CodeAdvEvalAttack(CodePlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(CodeAdvEvalAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
        self.target_instance = target_instance

    def get_init_prompt(self, item):
        return super().get_init_prompt(item)

    def get_prompt_with_source(self, source):
        return super().get_prompt_with_source(source)
     
    def att_model_generate(self, prompt, item):
        return att_gen_prompt(self, item)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass


class TranslationAdvEvalAttack(JudgeAttacker):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        self.result = None
        self.evaluator = benchmark_evaluator
        self.attack_instance = attack_instance
        self.target_instance = target_instance
        self.args = args
        self.is_system_should_optimize = False
        self.custom_injections = self.args.attack_custom_injections or {}
    
    def get_init_prompt(self, item):
        return get_translation_pure_prompt(self.args)

    def get_prompt_with_source(self, source):
        return get_translation_init_prompt(source)
     
    def att_model_generate(self, prompt, item):
        return att_gen_prompt(self, item)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass

from modules.attack.base_attack import JudgeAttacker
from modules.attack.plain import CodePlain, MathPlain, KnowledgePlain, SummarizationPlain
from modules.data import modify_model_generated
from utils.prompt import (
    get_translation_init_prompt,
    get_translation_pure_prompt, extract_text
)

def att_gen_prompt(self, prompt):
    if self.model_generated != None:
        content = self.model_generated
        content = extract_text(content)
    else:
        content = self.attack_instance.model_generate(prompt)
        content = extract_text(content)
        modify_model_generated(self.cache_data_path, self.cache_data_index, content)
    return content * 128


class TranslationLongAttack(JudgeAttacker):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        self.result = None
        self.evaluator = benchmark_evaluator
        self.attack_instance = attack_instance
        self.args = args
        self.is_system_should_optimize = False
        self.custom_injections = self.args.attack_custom_injections or {}
    
    def get_init_prompt(self, item):
        return get_translation_pure_prompt(self.args)

    def get_prompt_with_source(self, source):
        return get_translation_init_prompt(source)
     
    def att_model_generate(self, prompt, item):
        prompt = prompt.replace("##SOURCE_LAN##", self.args.source)\
                .replace("##TARGET_LAN##", self.args.target)
        return att_gen_prompt(self, prompt)
        
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass   
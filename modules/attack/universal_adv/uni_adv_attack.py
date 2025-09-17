from modules.attack.plain import (
    CodePlain, MathPlain, KnowledgePlain,
    SummarizationPlain, TranslationPlain
)
from modules.data import modify_model_generated
from utils.prompt import (
    extract_text
)
from modules.attack.universal_adv.utils import (
    get_attack_phrase_score, get_attack_phrase_pair
)
from loguru import logger
def adv_suffix_phrase(attack_instance, args):
    if args.judge == "score":
        return get_attack_phrase_score(attack_instance, args)
    elif args.judge == "pairwise":
        return get_attack_phrase_pair(attack_instance, args)
    else:
        default = " ;"
        return default

def att_gen_prompt(self, prompt):
    # logger.info(f'self.model_generated:{self.model_generated}')
    if self.model_generated != None:
        content = self.model_generated
        content = extract_text(content)
    else:
        content = self.attack_instance.model_generate(prompt)
        content = extract_text(content)
        modify_model_generated(self.cache_data_path, self.cache_data_index, content)
    return content + ' ' + adv_suffix_phrase(self.attack_instance, self.args)

def run_prompt_suffix(self):
    content = self.prompt
    return content + ' ' + adv_suffix_phrase(self.attack_instance, self.args)

class SummarizationUniAttack(SummarizationPlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(SummarizationUniAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
    
    def get_init_prompt(self, item):
        return super().get_init_prompt(item)

    def get_prompt_with_source(self, source):
        return super().get_prompt_with_source(source)
     
    def att_model_generate(self, prompt, item):
        return att_gen_prompt(self, prompt)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass

class CodeUniAttack(CodePlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(CodeUniAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
    
    def get_init_prompt(self, item):
        return super().get_init_prompt(item)

    def get_prompt_with_source(self, source):
        return super().get_prompt_with_source(source)
     
    def att_model_generate(self, prompt, item):
        return att_gen_prompt(self, prompt)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass

class MathUniAttack(MathPlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(MathUniAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
    
    def get_init_prompt(self, item):
        return super().get_init_prompt(item)

    def get_prompt_with_source(self, source):
        return super().get_prompt_with_source(source)
     
    def att_model_generate(self, prompt, item):
        return super().att_model_generate(prompt, item)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        self.prompt = run_prompt_suffix(self)
        return
    
class KnowledgeUniAttack(KnowledgePlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(KnowledgeUniAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
    
    def get_init_prompt(self, item):
        return super().get_init_prompt(item)

    def get_prompt_with_source(self, source):
        return super().get_prompt_with_source(source)
     
    def att_model_generate(self, prompt, item):
        return super().att_model_generate(prompt, item)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        self.prompt = run_prompt_suffix(self)
        return

class TranslationUniAttack(TranslationPlain):
    def __init__(self,benchmark_evaluator,attack_instance, target_instance,args):
        super(TranslationUniAttack, self).__init__(benchmark_evaluator,attack_instance, target_instance,args)
    
    def get_init_prompt(self, item):
        return super().get_init_prompt(item)

    def get_prompt_with_source(self, source):
        return super().get_prompt_with_source(source)
     
    def att_model_generate(self, prompt, item):
        prompt = prompt.replace("##SOURCE_LAN##", self.args.source)\
                .replace("##TARGET_LAN##", self.args.target)
        return att_gen_prompt(self, prompt)
    
    def get_bench_score(self, item, reference, candidate):
        return self.evaluator.meta_evaluate(item, reference, candidate)
    
    def run(self, item):
        pass
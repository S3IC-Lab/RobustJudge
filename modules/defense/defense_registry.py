# defense_registry.py
from collections import defaultdict


class DefenseRegistry:
    _registry = defaultdict(dict)  # 结构: {task_type: {defense_name: defense_class}}

    @classmethod
    def register(cls, task_type: str, defense_name: str):
        """Register a defense class for a specific task type."""
        print(f"#####register: task type is {task_type} and defense is {defense_name}")
        def decorator(defense_cls):
            cls._registry[task_type][defense_name] = defense_cls
            return defense_cls
        return decorator

    @classmethod
    def get_defense(cls, task_type: str, defense_name: str):
        """Get a defense class for a specific task type."""
        defense_cls = cls._registry[task_type].get(defense_name)
        print(f"defense class: {defense_cls}")
        if not defense_cls:
            print(f"task: {task_type}, defense name:{defense_name}")
            raise ValueError(f"The {defense_name} has not been supported in task: {task_type}")
        return defense_cls
    
#=====Register none defence here====#
from modules.defense.nonedefense import NoneDefense
DefenseRegistry.register("translation", "none")(NoneDefense)
DefenseRegistry.register("code_translation", "none")(NoneDefense)
DefenseRegistry.register("code_generation", "none")(NoneDefense)
DefenseRegistry.register("code_summarization", "none")(NoneDefense)
DefenseRegistry.register("math", "none")(NoneDefense)
DefenseRegistry.register("reasoning", "none")(NoneDefense)
DefenseRegistry.register("knowledge", "none")(NoneDefense)
DefenseRegistry.register("summarization", "none")(NoneDefense)

#=====Register retokenization defence here====#
from modules.defense.retokenization import RetokenizationDefense
DefenseRegistry.register("translation", "retokenization")(RetokenizationDefense)
DefenseRegistry.register("code_translation", "retokenization")(RetokenizationDefense)
DefenseRegistry.register("code_generation", "retokenization")(RetokenizationDefense)
DefenseRegistry.register("code_summarization", "retokenization")(RetokenizationDefense)
DefenseRegistry.register("math", "retokenization")(RetokenizationDefense)
DefenseRegistry.register("reasoning", "retokenization")(RetokenizationDefense)
DefenseRegistry.register("knowledge", "retokenization")(RetokenizationDefense)
DefenseRegistry.register("summarization", "retokenization")(RetokenizationDefense)

#=====Register delimiter defence here====#
from modules.defense.delimiter import DelimiterDefense
DefenseRegistry.register("translation", "delimiter")(DelimiterDefense)
DefenseRegistry.register("code_translation", "delimiter")(DelimiterDefense)
DefenseRegistry.register("code_generation", "delimiter")(DelimiterDefense)
DefenseRegistry.register("code_summarization", "delimiter")(DelimiterDefense)
DefenseRegistry.register("math", "delimiter")(DelimiterDefense)
DefenseRegistry.register("reasoning", "delimiter")(DelimiterDefense)
DefenseRegistry.register("knowledge", "delimiter")(DelimiterDefense)
DefenseRegistry.register("summarization", "delimiter")(DelimiterDefense)

#=====Register sandwich defence here====#
from modules.defense.sandwich import SandwichDefense
DefenseRegistry.register("translation", "sandwich")(SandwichDefense)
DefenseRegistry.register("code_translation", "sandwich")(SandwichDefense)
DefenseRegistry.register("code_generation", "sandwich")(SandwichDefense)
DefenseRegistry.register("code_summarization", "sandwich")(SandwichDefense)
DefenseRegistry.register("math", "sandwich")(SandwichDefense)
DefenseRegistry.register("reasoning", "sandwich")(SandwichDefense)
DefenseRegistry.register("knowledge", "sandwich")(SandwichDefense)
DefenseRegistry.register("summarization", "sandwich")(SandwichDefense)

#=====Register preprompt defence here====#
from modules.defense.preprompt import PrepromptDefense
DefenseRegistry.register("translation", "preprompt")(PrepromptDefense)
DefenseRegistry.register("code_translation", "preprompt")(PrepromptDefense)
DefenseRegistry.register("code_generation", "preprompt")(PrepromptDefense)
DefenseRegistry.register("code_summarization", "preprompt")(PrepromptDefense)
DefenseRegistry.register("math", "preprompt")(PrepromptDefense)
DefenseRegistry.register("reasoning", "preprompt")(PrepromptDefense)
DefenseRegistry.register("knowledge", "preprompt")(PrepromptDefense)
DefenseRegistry.register("summarization", "preprompt")(PrepromptDefense)

#=====Register ppl defence here====#
from modules.defense.ppl import PplDefense
DefenseRegistry.register("translation", "ppl")(PplDefense)
DefenseRegistry.register("code_translation", "ppl")(PplDefense)
DefenseRegistry.register("code_generation", "ppl")(PplDefense)
DefenseRegistry.register("code_summarization", "ppl")(PplDefense)
DefenseRegistry.register("math", "ppl")(PplDefense)
DefenseRegistry.register("reasoning", "ppl")(PplDefense)
DefenseRegistry.register("knowledge", "ppl")(PplDefense)
DefenseRegistry.register("summarization", "ppl")(PplDefense)

#=====Register winppl defence here====#
from modules.defense.winppl import WinpplDefense
DefenseRegistry.register("translation", "winppl")(WinpplDefense)
DefenseRegistry.register("code_translation", "winppl")(WinpplDefense)
DefenseRegistry.register("code_generation", "winppl")(WinpplDefense)
DefenseRegistry.register("code_summarization", "winppl")(WinpplDefense)
DefenseRegistry.register("math", "winppl")(WinpplDefense)
DefenseRegistry.register("reasoning", "winppl")(WinpplDefense)
DefenseRegistry.register("knowledge", "winppl")(WinpplDefense)
DefenseRegistry.register("summarization", "winppl")(WinpplDefense)

#=====Register naivellm defence here====#
from modules.defense.naivellm import NaivellmDefense
DefenseRegistry.register("translation", "naivellm")(NaivellmDefense)
DefenseRegistry.register("code_translation", "naivellm")(NaivellmDefense)
DefenseRegistry.register("code_generation", "naivellm")(NaivellmDefense)
DefenseRegistry.register("code_summarization", "naivellm")(NaivellmDefense)
DefenseRegistry.register("math", "naivellm")(NaivellmDefense)
DefenseRegistry.register("reasoning", "naivellm")(NaivellmDefense)
DefenseRegistry.register("knowledge", "naivellm")(NaivellmDefense)
DefenseRegistry.register("summarization", "naivellm")(NaivellmDefense)

#=====Register pto defence here====#
from modules.defense.pto import PTODefense
DefenseRegistry.register("translation", "pto")(PTODefense)

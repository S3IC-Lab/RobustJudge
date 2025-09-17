# attack_registry.py
from collections import defaultdict
from modules.attack.manual_prompt_injection import (
    TranslationNaiveAttack, TranslationContextIgnore,
    TranslationEscapeCharacters, TranslationFakeCompletion,
    TranslationCombinedAttack, TranslationFakeReasoning,
    TranslationEmptyAttack, TranslationLongAttack,

    CodeNaiveAttack, CodeContextIgnore,
    CodeEscapeCharacters, CodeFakeCompletion,
    CodeCombinedAttack, CodeFakeReasoning,
    CodeEmptyAttack,

    MathNaiveAttack, MathContextIgnore,
    MathEscapeCharacters, MathFakeCompletion,
    MathCombinedAttack, MathFakeReasoning,
    MathEmptyAttack,

    KnowledgeNaiveAttack, KnowledgeContextIgnore,
    KnowledgeEscapeCharacters, KnowledgeCombinedAttack,
    KnowledgeFakeReasoning, KnowledgeFakeCompletion,
    KnowledgeEmptyAttack,

    SummarizationNaiveAttack, SummarizationContextIgnore,
    SummarizationEscapeCharacters, SummarizationCombinedAttack,
    SummarizationFakeReasoning, SummarizationFakeCompletion,
    SummarizationEmptyAttack
)
from modules.attack.prompt_optimization import (
TranslationPromptOptimization
)
from modules.attack.plain import (
    TranslationPlain, CodePlain, MathPlain, KnowledgePlain, SummarizationPlain
)
from modules.attack.adv_eval import (
    TranslationAdvEvalAttack, SummarizationAdvEvalAttack, CodeAdvEvalAttack
)
from modules.attack.cheating_llm import (
    TranslationCheatingAttack, SummarizationCheatingAttack, CodeCheatingAttack,
    KnowledgeCheatingAttack, MathCheatingAttack
)
from modules.attack.jailbreak import (
    TranslationGCGAttack, KnowledgeGCGAttack, MathGCGAttack,
    SummarizationGCGAttack, CodeGCGAttack,

    TranslationAutoDANAttack, KnowledgeAutoDANAttack,
    MathAutoDANAttack, SummarizationAutoDANAttack, CodeAutoDANAttack,

    TranslationPairAttack, SummarizationPairAttack, CodePairAttack,

    TranslationTapAttack, SummarizationTapAttack, CodeTapAttack
)
from modules.attack.universal_adv import (
    TranslationUniAttack, CodeUniAttack, 
    MathUniAttack, KnowledgeUniAttack, SummarizationUniAttack
)

class AttackRegistry:
    _registry = defaultdict(dict)  # 结构: {task_type: {attack_name: attack_class}}

    @classmethod
    def register(cls, task_type: str, attack_name: str):
        """Register an attack class for a specific task type."""
        print(f"#####register: task type is {task_type} and attack is {attack_name}")
        def decorator(attack_cls):
            cls._registry[task_type][attack_name] = attack_cls
            return attack_cls
        return decorator

    @classmethod
    def get_attack(cls, task_type: str, attack_name: str):
        """Get an attack class for a specific task type."""
        attack_cls = cls._registry[task_type].get(attack_name)
        print(f"attack class: {attack_cls}")
        if not attack_cls:
            print(f"task: {task_type}, attack name:{attack_name}")
            raise ValueError(f"The {attack_name} attack has not been supported in task: {task_type}")
        return attack_cls
    
#=====(Translation Task)Register manual prompt injection strategies here====#
AttackRegistry.register("translation", "none")(TranslationPlain)
AttackRegistry.register("translation", "naive")(TranslationNaiveAttack)
AttackRegistry.register("translation", "ignore")(TranslationContextIgnore)
AttackRegistry.register("translation", "escape")(TranslationEscapeCharacters)
AttackRegistry.register("translation", "completion")(TranslationFakeCompletion)
AttackRegistry.register("translation", "combined")(TranslationCombinedAttack)
AttackRegistry.register("translation", "reasoning")(TranslationFakeReasoning)
AttackRegistry.register("translation", "empty")(TranslationEmptyAttack)
AttackRegistry.register("translation", "long")(TranslationLongAttack)


#=====(Code Task)Register manual-prompt-injection strategies here===========#
AttackRegistry.register("code_generation", "none")(CodePlain)
AttackRegistry.register("code_summarization", "none")(CodePlain)
AttackRegistry.register("code_translation", "none")(CodePlain)

AttackRegistry.register("code_generation", "naive")(CodeNaiveAttack)
AttackRegistry.register("code_summarization", "naive")(CodeNaiveAttack)
AttackRegistry.register("code_translation", "naive")(CodeNaiveAttack)

AttackRegistry.register("code_generation", "ignore")(CodeContextIgnore)
AttackRegistry.register("code_summarization", "ignore")(CodeContextIgnore)
AttackRegistry.register("code_translation", "ignore")(CodeContextIgnore)

AttackRegistry.register("code_generation", "escape")(CodeEscapeCharacters)
AttackRegistry.register("code_summarization", "escape")(CodeEscapeCharacters)
AttackRegistry.register("code_translation", "escape")(CodeEscapeCharacters)

AttackRegistry.register("code_generation", "completion")(CodeFakeCompletion)
AttackRegistry.register("code_summarization", "completion")(CodeFakeCompletion)
AttackRegistry.register("code_translation", "completion")(CodeFakeCompletion)

AttackRegistry.register("code_generation", "combined")(CodeCombinedAttack)
AttackRegistry.register("code_summarization", "combined")(CodeCombinedAttack)
AttackRegistry.register("code_translation", "combined")(CodeCombinedAttack)

AttackRegistry.register("code_generation", "reasoning")(CodeFakeReasoning)
AttackRegistry.register("code_summarization", "reasoning")(CodeFakeReasoning)
AttackRegistry.register("code_translation", "reasoning")(CodeFakeReasoning)

AttackRegistry.register("code_generation", "empty")(CodeEmptyAttack)
AttackRegistry.register("code_summarization", "empty")(CodeEmptyAttack)
AttackRegistry.register("code_translation", "empty")(CodeEmptyAttack)

#=====(Math Task)Register manual-prompt-injection strategies here===========#
AttackRegistry.register("math", "none")(MathPlain)
AttackRegistry.register("math", "naive")(MathNaiveAttack)
AttackRegistry.register("math", "ignore")(MathContextIgnore)
AttackRegistry.register("math", "escape")(MathEscapeCharacters)
AttackRegistry.register("math", "completion")(MathFakeCompletion)
AttackRegistry.register("math", "combined")(MathCombinedAttack)
AttackRegistry.register("math", "reasoning")(MathFakeReasoning)
AttackRegistry.register("math", "empty")(MathEmptyAttack)

#=====(Reasoning Task)Register manual-prompt-injection strategies here===========#
AttackRegistry.register("reasoning", "none")(MathPlain)
AttackRegistry.register("reasoning", "naive")(MathNaiveAttack)
AttackRegistry.register("reasoning", "ignore")(MathContextIgnore)
AttackRegistry.register("reasoning", "escape")(MathEscapeCharacters)
AttackRegistry.register("reasoning", "combined")(MathCombinedAttack)
AttackRegistry.register("reasoning", "reasoning")(MathFakeReasoning)
AttackRegistry.register("reasoning", "completion")(MathFakeCompletion)
AttackRegistry.register("reasoning", "empty")(MathEmptyAttack)

#=====(Knowledge Task)Register manual-prompt-injection strategies here===========#
AttackRegistry.register("knowledge", "none")(KnowledgePlain)
AttackRegistry.register("knowledge", "naive")(KnowledgeNaiveAttack)
AttackRegistry.register("knowledge", "ignore")(KnowledgeContextIgnore)
AttackRegistry.register("knowledge", "escape")(KnowledgeEscapeCharacters)
AttackRegistry.register("knowledge", "combined")(KnowledgeCombinedAttack)
AttackRegistry.register("knowledge", "reasoning")(KnowledgeFakeReasoning)
AttackRegistry.register("knowledge", "completion")(KnowledgeFakeCompletion)
AttackRegistry.register("knowledge", "empty")(KnowledgeEmptyAttack)

#=====(SummarizationPlain Task)Register manual-prompt-injection strategies here===========#
AttackRegistry.register("summarization", "none")(SummarizationPlain)
AttackRegistry.register("summarization", "naive")(SummarizationNaiveAttack)
AttackRegistry.register("summarization", "ignore")(SummarizationContextIgnore)
AttackRegistry.register("summarization", "escape")(SummarizationEscapeCharacters)
AttackRegistry.register("summarization", "combined")(SummarizationCombinedAttack)
AttackRegistry.register("summarization", "reasoning")(SummarizationFakeReasoning)
AttackRegistry.register("summarization", "completion")(SummarizationFakeCompletion)
AttackRegistry.register("summarization", "empty")(SummarizationEmptyAttack)


# ==================== Register prompt optimization method here ====================#
AttackRegistry.register("translation", "naive_po")(TranslationPromptOptimization)

# ==================== Register adv method here ====================#
AttackRegistry.register("translation", "adv")(TranslationAdvEvalAttack)
AttackRegistry.register("summarization", "adv")(SummarizationAdvEvalAttack)
AttackRegistry.register("code_generation", "adv")(CodeAdvEvalAttack)
AttackRegistry.register("code_summarization", "adv")(CodeAdvEvalAttack)
AttackRegistry.register("code_translation", "adv")(CodeAdvEvalAttack)

# ==================== Register cheating method here ====================#
AttackRegistry.register("translation", "cheating")(TranslationCheatingAttack)
AttackRegistry.register("summarization", "cheating")(SummarizationCheatingAttack)
AttackRegistry.register("code_generation", "cheating")(CodeCheatingAttack)
AttackRegistry.register("code_summarization", "cheating")(CodeCheatingAttack)
AttackRegistry.register("code_translation", "cheating")(CodeCheatingAttack)
AttackRegistry.register("math", "cheating")(MathCheatingAttack)
AttackRegistry.register("reasoning", "cheating")(MathCheatingAttack)
AttackRegistry.register("knowledge", "cheating")(KnowledgeCheatingAttack)

# ==================== Register jailbreak: gcg_attack method here ====================#
AttackRegistry.register("translation", "gcg")(TranslationGCGAttack)
AttackRegistry.register("knowledge", "gcg")(KnowledgeGCGAttack)
AttackRegistry.register("reasoning", "gcg")(MathGCGAttack)
AttackRegistry.register("math", "gcg")(MathGCGAttack)
AttackRegistry.register("summarization", "gcg")(SummarizationGCGAttack)
AttackRegistry.register("code_generation", "gcg")(CodeGCGAttack)
AttackRegistry.register("code_summarization", "gcg")(CodeGCGAttack)
AttackRegistry.register("code_translation", "gcg")(CodeGCGAttack)

# ==================== Register jailbreak: autodan_attack here ====================#
AttackRegistry.register("translation", "autodan")(TranslationAutoDANAttack)
AttackRegistry.register("knowledge", "autodan")(KnowledgeAutoDANAttack)
AttackRegistry.register("reasoning", "autodan")(MathAutoDANAttack)
AttackRegistry.register("math", "autodan")(MathAutoDANAttack)
AttackRegistry.register("summarization", "autodan")(SummarizationAutoDANAttack)
AttackRegistry.register("code_generation", "autodan")(CodeAutoDANAttack)
AttackRegistry.register("code_summarization", "autodan")(CodeAutoDANAttack)
AttackRegistry.register("code_translation", "autodan")(CodeAutoDANAttack)

# ==================== Register jailbreak: pair_attack here ====================#
AttackRegistry.register("translation", "pair")(TranslationPairAttack)
AttackRegistry.register("summarization", "pair")(SummarizationPairAttack)
AttackRegistry.register("code_generation", "pair")(CodePairAttack)
AttackRegistry.register("code_summarization", "pair")(CodePairAttack)
AttackRegistry.register("code_translation", "pair")(CodePairAttack)

# ==================== Register jailbreak: tap_attack here ====================#
AttackRegistry.register("translation", "tap")(TranslationTapAttack)
AttackRegistry.register("summarization", "tap")(SummarizationTapAttack)
AttackRegistry.register("code_generation", "tap")(CodeTapAttack)
AttackRegistry.register("code_summarization", "tap")(CodeTapAttack)
AttackRegistry.register("code_translation", "tap")(CodeTapAttack)

# ==================== Register uni_adv_attack method here ====================#
AttackRegistry.register("translation", "uni")(TranslationUniAttack)
AttackRegistry.register("summarization", "uni")(SummarizationUniAttack)
AttackRegistry.register("code_generation", "uni")(CodeUniAttack)
AttackRegistry.register("code_summarization", "uni")(CodeUniAttack)
AttackRegistry.register("code_translation", "uni")(CodeUniAttack)
AttackRegistry.register("math", "uni")(MathUniAttack)
AttackRegistry.register("reasoning", "uni")(MathUniAttack)
AttackRegistry.register("knowledge", "uni")(KnowledgeUniAttack)

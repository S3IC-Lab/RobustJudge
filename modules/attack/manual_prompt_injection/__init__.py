from modules.attack.manual_prompt_injection.naive_attack import (
    TranslationNaiveAttack,
    MathNaiveAttack,
    CodeNaiveAttack,
    KnowledgeNaiveAttack,
    SummarizationNaiveAttack
)

from modules.attack.manual_prompt_injection.context_ignore import (
    TranslationContextIgnore,
    MathContextIgnore,
    CodeContextIgnore,
    KnowledgeContextIgnore,
    SummarizationContextIgnore
)

from modules.attack.manual_prompt_injection.escape_characters import (
    TranslationEscapeCharacters,
    MathEscapeCharacters,
    CodeEscapeCharacters,
    KnowledgeEscapeCharacters,
    SummarizationEscapeCharacters
)

from modules.attack.manual_prompt_injection.fake_completion import (
    TranslationFakeCompletion,
    MathFakeCompletion,
    CodeFakeCompletion,
    SummarizationFakeCompletion,
    KnowledgeFakeCompletion
)

from modules.attack.manual_prompt_injection.combined_attack import (
    TranslationCombinedAttack,
    MathCombinedAttack,
    CodeCombinedAttack,
    KnowledgeCombinedAttack,
    SummarizationCombinedAttack
)

from modules.attack.manual_prompt_injection.fake_reasoning import (
    TranslationFakeReasoning,
    MathFakeReasoning,
    CodeFakeReasoning,
    KnowledgeFakeReasoning,
    SummarizationFakeReasoning
)

from modules.attack.manual_prompt_injection.empty_attack import (
    TranslationEmptyAttack,
    MathEmptyAttack,
    CodeEmptyAttack,
    KnowledgeEmptyAttack,
    SummarizationEmptyAttack
)

from modules.attack.manual_prompt_injection.long_suffix_attack import (
    TranslationLongAttack
)

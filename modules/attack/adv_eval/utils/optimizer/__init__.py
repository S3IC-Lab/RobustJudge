from .optimizer_chatgpt import (
    ChatGPTOptimizerDialog,
    ChatGPTOptimizerQA,
    ChatGPTOptimizerSum,
    ChatGPTOptimizerDialogSum,
    ChatGPTOptimizerTranslation
)

from .optimizer_attacker import (
    AttackerOptimizerTranslation,
    AttackerOptimizerCodeTrans,
    AttackerOptimizerCodeGen,
    AttackerOptimizerCodeSummary,
    AttackerOptimizerSummary
)

optimizer_list = {
    'optimizer_dialog_gpt4': ChatGPTOptimizerDialog,
    'optimizer_qa_gpt4': ChatGPTOptimizerQA,
    'optimizer_sum_gpt4': ChatGPTOptimizerSum,
    'optimizer_dialogsum_gpt4': ChatGPTOptimizerDialogSum,
    'optimizer_translation_gpt4': ChatGPTOptimizerTranslation,
    'optimizer_translation_attacker': AttackerOptimizerTranslation,
    'optimizer_summarization_attacker': AttackerOptimizerSummary,
    'optimizer_code_summarization_attacker': AttackerOptimizerCodeSummary,
    'optimizer_code_generation_attacker': AttackerOptimizerCodeGen,
    'optimizer_code_translation_attacker': AttackerOptimizerCodeTrans

}


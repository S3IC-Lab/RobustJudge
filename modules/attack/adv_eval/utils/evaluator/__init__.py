from .bertscore import BERTSCOREEvaluator
from .sacrebleu import SACREBLEUEvaluator
from .poe import PoEEvaluator
from .bleurt import BLEURTEvaluator
from .llmeval import LLMDialogEvaluator, LLMDialogSumEvaluator, LLMSumEvaluator, LLMQAEvaluator
from .unieval import DialogEvaluator as UniDialogEvaluator
from .unieval import SumEvaluator as UniSumEvaluator
from .bartscore import BARTSCOREEvaluator
from .rouge import ROUGEEvaluator
from .true import TrueEvaluator
from .chatgpt import (
    ChatGPTDialogEvaluator,
    ChatGPTQAEvaluator,
    ChatGPTSumEvaluator,
    ChatGPTDialogSumEvaluator
)

evaluator_list = {
    'rouge': ROUGEEvaluator,
    'true': TrueEvaluator,
    'bertscore': BERTSCOREEvaluator,
    'bartscore': BARTSCOREEvaluator,
    'sacrebleu': SACREBLEUEvaluator,
    'bleurt': BLEURTEvaluator,
    'poe': PoEEvaluator,
    'chatgpt_dialog': ChatGPTDialogEvaluator,
    'chatgpt_qa': ChatGPTQAEvaluator,
    'chatgpt_sum': ChatGPTSumEvaluator,
    'chatgpt_dialogsum': ChatGPTDialogSumEvaluator,
    'llm_dialog': LLMDialogEvaluator,
    'llm_dialogsum': LLMDialogSumEvaluator,
    'llm_sum': LLMSumEvaluator,
    'llm_qa': LLMQAEvaluator,
    'unieval_dialog': UniDialogEvaluator,
    'unieval_sum': UniSumEvaluator,
}

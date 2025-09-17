from modules.eval.registry import EvaluatorRegistry
from modules.eval.acc import AccuracyEvaluator
from modules.eval.asr import AsrEvaluator
from modules.eval.bleurt import BLEURTEvaluator
from modules.eval.llm_judge.judge_score import SdrEvaluator
from modules.eval.gpt_judge import GPTJudgeEvaluator
from modules.eval.codebleu import CodebleuEvaluator
from modules.eval.pre_judge import PreJudgeEvaluator

evaluator_list = {
    'translation': 'bleurt',
    'code_translation': 'codebleu',
    'code_generation': "codebleu",
    'code_summarization': 'bleurt',
    'math': 'pre_judge',
    'reasoning': 'pre_judge',
    'knowledge': 'pre_judge',
    'summarization': 'bleurt'
}

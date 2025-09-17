# evaluators/asr.py

from modules.eval.base_evaluator import BaseEvaluator
from modules.eval.registry import EvaluatorRegistry
from model.openai_api import chat_completion_openai_demo
from model import get_conversation_template

@EvaluatorRegistry.register("gpt_judge")
class GPTJudgeEvaluator(BaseEvaluator):
    """
    Evaluator that calculates the accuracy of the model.
    """

    def evaluate(self, prompt):
        conv = get_conversation_template("gpt-4o-2024-11-20")
    
        conv.append_message(conv.roles[0], prompt)

        output = chat_completion_openai_demo("gpt-4o-2024-11-20", conv, 0.4, 1024)
                
        try:
            judgment = output
        except Exception as e:
            print(f"Error extracting score for source")
            judgment = None
        return judgment
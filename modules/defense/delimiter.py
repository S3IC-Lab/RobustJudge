from modules.defense.base_defense import JudgeDefender
from loguru import logger
from utils.prompt import get_score_prompt, get_pair_prompt
from utils.common import extract_score
delimiter_prompts = {
    # https://github.com/liu00222/Open-Prompt-Injection/blob/main/OpenPromptInjection/apps/Application.py
    "translation": "(the translation below is delimited by the ```)\n",
    "code_translation": "(the translated code below is delimited by the ```)\n",
    "code_generation": "(the generated code below is delimited by the ```)\n",
    "code_summarization": "(the summarization of the code below is delimited by the ```)\n",
    "math": "(the solution below is delimited by the ```)\n",
    "reasoning": "(the solution below is delimited by the ```)\n",
    "knowledge": "(the answer below is delimited by the ```)\n",
    "summarization": "(the summarization below is delimited by the ```)\n"
}

class DelimiterDefense(JudgeDefender):
    def __init__(self, attack_instance, defense_instance, args):
        super().__init__(attack_instance, defense_instance, args)
    
    def make_delimited_prompt(self, item):
        args = self.args
        delimitered_data = delimiter_prompts.get(args.task) + '```\n' + item['malicious answer'] + '\n```\n'
        logger.info(f"delimitered_data:{delimitered_data}")
        if 'id' not in item:
            return delimitered_data
        if args.judge == 'score':
            self.score_prompt = get_score_prompt(item['source'], delimitered_data, args)
        else:
            # pairwise judge
            self.pairwise_prompt_A = get_pair_prompt(item['source'], delimitered_data, item['target'], args)
            self.pairwise_prompt_B = get_pair_prompt(item['source'], item['target'], delimitered_data, args)
        return delimitered_data

    def run(self):
        data = self.load_pre_result()
        result = data
        for i, item in enumerate(data):
            if 'id' not in item:
                continue
            result[i]['defense_content'] = ''
            result[i]['defense_delimited_content'] = self.make_delimited_prompt(item)

            self.deal_with_defense_result(result, i, item)
        
        self.output_defense_result(result)
        return

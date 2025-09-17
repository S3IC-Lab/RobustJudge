from modules.defense.base_defense import JudgeDefender
from .ppl import calculate_perplexity
from nltk.tokenize import sent_tokenize
from loguru import logger


class WinpplDefense(JudgeDefender):
    def __init__(self, attack_instance, defense_instance, args):
        super().__init__(attack_instance, defense_instance, args)
        self.win_size = 5

    def cal_win_ppl(self, text, model, tokenizer):
        sentences = sent_tokenize(text)
        windows = [sentences[i:i+self.win_size] for i in range(0, len(sentences), self.win_size)]
        max_ppl = 0
        logger.info(f'windows: {windows}')
        for item in windows:
            win_text = ''.join(item)
            # logger.info(f'window text: {win_text}')
            ppl = calculate_perplexity(win_text, model, tokenizer)
            if ppl > max_ppl:
                max_ppl = ppl
        return max_ppl


    def run(self):
        data = self.load_pre_result()
        result = data
        model = self.attack_instance.model
        tokenizer = self.attack_instance.tokenizer
        for i, item in enumerate(data):
            if 'id' not in item:
                # last item in the list
                continue
            adv_text = item['malicious answer']
            if adv_text == '':
                adv_ppl = float('inf')  # Assign infinity if the text is empty
            else:
                adv_ppl = self.cal_win_ppl(adv_text, model, tokenizer)

            threshold_text = item['target']
            threshold_ppl = calculate_perplexity(threshold_text, model, tokenizer)
            
            result[i]['defense_adv_ppl'] = adv_ppl
            result[i]['defense_threshold_ppl'] = threshold_ppl
            result[i]['defense_result'] = 'defense fail'

            if adv_ppl > threshold_ppl:
                result[i]['defense_result'] = 'defense success'
        
        self.output_defense_result(result)
        return
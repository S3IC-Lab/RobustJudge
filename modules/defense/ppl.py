import torch
import torch.nn as nn
import math
from modules.defense.base_defense import JudgeDefender
from loguru import logger

def calculate_perplexity(text, model, tokenizer, max_length=4096):
    encodings = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
    input_ids = encodings.input_ids.to(model.device)

    log_probs = []
    for i in range(1, input_ids.size(1)):
        # input the first i tokens and predict the i-th token
        inputs = input_ids[:, :i]
        with torch.no_grad():
            outputs = model(inputs)
            # get logits for the i-th token
            logits = outputs.logits[0, -1, :]
            # calculate the log probability of the true token
            log_prob = torch.log_softmax(logits, dim=-1)[input_ids[0, i]]
            log_probs.append(log_prob)
    
    # calculate the mean cross-entropy loss
    nll = -torch.mean(torch.stack(log_probs))
    ppl = torch.exp(nll)
    return ppl.item()

class PplDefense(JudgeDefender):
    def __init__(self, attack_instance, defense_instance, args):
        super().__init__(attack_instance, defense_instance, args)

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
                adv_ppl = calculate_perplexity(adv_text, model, tokenizer)

            threshold_text = item['target']
            threshold_ppl = calculate_perplexity(threshold_text, model, tokenizer)
            
            result[i]['defense_adv_ppl'] = adv_ppl
            result[i]['defense_threshold_ppl'] = threshold_ppl
            result[i]['defense_result'] = 'defense fail'

            if adv_ppl > threshold_ppl:
                result[i]['defense_result'] = 'defense success'
        
        self.output_defense_result(result)
        return
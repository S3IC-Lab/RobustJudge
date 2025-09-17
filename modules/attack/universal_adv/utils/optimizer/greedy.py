import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import os
from abc import ABC, abstractmethod
from .common_utils import next_dir, get_cache_name
from modules.data import replace_invalid_characters
from utils.prompt import get_pair_prompt, get_score_prompt
from model import get_conversation_template
from transformers import AutoModelForCausalLM
from types import SimpleNamespace
from loguru import logger
import re

class BaseGreedyAttacker(ABC):
    def __init__(self, attack_instance, word_list, args):
        '''
            attack_instance: instance of the attack model
            word_list: list of words to be used for adversarial phrase
        '''
        self.attack_instance = attack_instance
        self.seen_systems = [2,4]
        self.word_list = word_list
        self.args = args
        self.model_id = attack_instance.model_id
        self.tokenizer = attack_instance.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.conv = get_conversation_template(self.model_id)
        self.find_batch_size = 2000
        
    def setup_label_words(self, label_words):
        label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]
        self.label_ids_tensor = torch.tensor(label_ids, dtype=torch.long)
        return label_ids

    def next_word_score(self, data, curr_adv_phrase, cache_path, start_index, num_attack_phrase_words):
        '''
            curr_adv_phrase: current universal adversarial phrase
            Returns the assessment score for each word in word list as next uni adv word
        '''
        args = self.args
        # check for cache
        pos_num = len(curr_adv_phrase.split(' '))+1 if curr_adv_phrase != '' else 1
        dir_name = get_cache_name(args, pos_num)
        path = next_dir(cache_path, dir_name)
        if not os.path.isdir(path):
            # create dirs
            os.makedirs(path, exist_ok=True)
        fpath_prev = f'{path}/prev.json'
        fpath_scores = f'{path}/scores.json'
        if os.path.isfile(fpath_prev):
            with open(fpath_prev, 'r') as f:
                prev = json.load(f)
            with open(fpath_scores, 'r') as f:
                word_2_score = json.load(f)
            return prev, word_2_score
        # logger.info(f"---sample_evaluate_uni_attack_seen start---")
        score_no_attack = self.sample_evaluate_uni_attack_seen(data, curr_adv_phrase)
        logger.info(f"score_no_attack:{score_no_attack}")
        score_adv_phrase = score_no_attack
        word_2_score = {}
        cnt = 0
        for word in tqdm(self.word_list):
            if cnt < start_index:
                cnt = cnt + 1
                continue
            if cnt > start_index + self.find_batch_size \
                and len(curr_adv_phrase.split(' ')) >= num_attack_phrase_words:
                break
            if curr_adv_phrase == '':
                adv_phrase = word + '.'
            else:
                adv_phrase = curr_adv_phrase + ' ' + word + '.'
            score = self.sample_evaluate_uni_attack_seen(data, adv_phrase)
            # logger.info(f"[{cnt}]word:{word}, score:{score}")
            word_2_score[word] = score
            if score > score_adv_phrase:
                curr_adv_phrase = adv_phrase.rstrip('.')
                score_adv_phrase = score
                logger.info(f"curr_adv_phrase:{curr_adv_phrase}, score_adv_phrase:{score_adv_phrase}")
            cnt = cnt + 1
        # cache
        with open(fpath_prev, 'w') as f:
            prev = {'prev-adv-phrase': curr_adv_phrase, 'score': score_adv_phrase,
                    'start_index': start_index, 'end_index': cnt}
            json.dump(prev, f, ensure_ascii=False, indent=4)
        with open(fpath_scores, 'w') as f:
            json.dump(word_2_score, f, ensure_ascii=False, indent=4)
        
        return prev, word_2_score

class GreedyComparativeAttacker(BaseGreedyAttacker):
    def __init__(self, attack_instance, word_list, args):
        super().__init__(attack_instance, word_list, args)
        self.pair_label=['A', 'B']
        self.label_ids = self.setup_label_words(self.pair_label)
    
    def calculate_prob(self, prompt, expect_answer):
        content, top_logprobs = self.attack_instance.model_generate_loss(
            prompt,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=True,
            top_logprobs=20,
            seed=0)
        for i in top_logprobs:
            if i.token == expect_answer:
                return math.exp(i.logprob)
        return 0.0
    
    def sample_evaluate_uni_attack_seen(self, data, adv_phrase=''):
        result = 0
        for sample in data:
            item_source = sample['source']
            item_a = sample['model_generated']
            item_b = sample['target']
            if adv_phrase != '':
                item_a = item_a + ' ' + adv_phrase
            
            with torch.no_grad():
                #A
                prompt = get_pair_prompt(item_source, item_a, item_b, self.args)
                prob1 = self.calculate_prob(prompt, 'A')
                #B
                prompt = get_pair_prompt(item_source, item_b, item_a, self.args)
                prob2 = self.calculate_prob(prompt, 'B')
                
                prob_i_better = 0.5*(prob1+prob2)
                result += prob_i_better

        return result/len(data)


class GreedyAbsoluteAttacker(BaseGreedyAttacker):
    def __init__(self, attack_instance, word_list, args):
        super().__init__(attack_instance, word_list, args)
        self.word_list = word_list
        self.score_labels= [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        self.scores = torch.LongTensor([int(i) for i in self.score_labels])
        self.label_words = [str(i) for i in self.score_labels]
        self.label_ids = self.setup_label_words(self.label_words)

    def sample_evaluate_uni_attack_seen(self, data, adv_phrase=''):
        result = 0
        for sample in data:
            source = sample['source']
            input = sample['model_generated']
            if adv_phrase != '':
                input = input + ' ' + adv_phrase
            prompt = get_score_prompt(source, input, self.args)
            
            judge_score = self.attack_instance.model_generate(prompt)
            # match the format:.5, 5., etc.
            float_pattern = r'[-+]?(\d+([.]\d*)?|[.]\d+)(?:[eE][-+]?\d+)?'
            score_match = re.search(float_pattern, judge_score)
            if not score_match:
                return 0.0
            
            raw_float = score_match.group(0)
            score = float(raw_float)
            result += score

        return result/len(data)



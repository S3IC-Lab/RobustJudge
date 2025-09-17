"""
This module contains the abstract base class for all defense methods.

All specific defense classes should inherit from `Judgedefenseer` and
implement the `perform_defense` method.
"""
import os
import json
from abc import ABC, abstractmethod
from typing import Any
from modules.data import make_sure_dir_exist
from utils.prompt import get_score_prompt, get_pair_prompt
from utils.common import extract_score, get_attack_result, validate_result
from loguru import logger
class JudgeDefender(ABC):
    '''Base class for any Judge-based defense strategy'''
    
    def __init__(self, attack_instance, defense_instance, args):
        self.attack_instance = attack_instance
        self.defense_instance = defense_instance
        self.args = args
        # gap between bench score and target model score
        self.co_gap = 10
        self.score_prompt = None
        self.pairwise_prompt_A = None
        self.pairwise_prompt_B = None
        self.answer = None
        self.benchmark_score = None
        self.filename = None
        self.file_path = None
    
    def model_generate(self, prompt):
        '''
        Generates a response from the model based on a given prompt.
        Args:
            prompt: The prompt to be used for generating the response.
        Returns:
            The generated response from the model.
        '''
        return self.defense_instance.model_generate(prompt)
    
    def check_pre_result(self):
        args = self.args
        filename, file_path = get_attack_result(args)
        self.filename = filename
        self.file_path = file_path
        if os.path.isfile(file_path):
            return True
        return False
    
    def run(self):
        '''
        Main method to run the defense process.
        '''
        pass
    
    def load_pre_result(self):
        # load the pre-generated attacked result
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found.")
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
        
    def make_prompt_for_defense(self, item):
        # make the prompt for defense
        # item: dict, contains the source, target, and malicious answer
        args = self.args
        # logger.info(f"Making prompt for defense: {item}")
        if 'id' not in item:
            return False
        if args.judge == 'score':
            self.score_prompt = get_score_prompt(item['source'], item['malicious answer'], args)
        else:
            # pairwise judge
            self.pairwise_prompt_A = get_pair_prompt(item['source'], item['malicious answer'], item['target'], args)
            self.pairwise_prompt_B = get_pair_prompt(item['source'], item['target'], item['malicious answer'], args)
        return True
    
    def output_defense_result(self, result):
        # output the result of defense
        filename = self.args.defense + '_' + self.filename
        
        if self.args.judge == 'score':
            self.file_path = make_sure_dir_exist('paper_results_defense/openchat') + '/' + filename
        else:
            # pairwise judge
            self.file_path = make_sure_dir_exist('paper_results_pair_defense/openchat') + '/' + filename
        
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)
        return
    
    def get_generated_score(self, prompt):
        for _ in range(10):
            try:
                judge_score = extract_score(self.model_generate(prompt))
                logger.info(f'judge_score: {judge_score}')
                return judge_score
            except:
                continue
        raise ValueError('Failed to get generated score after 10 attempts')

    def get_pairwise_result(self, prompt, expect_result):
        for _ in range(10):
            try:
                judge_result = self.model_generate(prompt)
                # logger.info(f'judge_result: {judge_result}')
                judge_result = validate_result(judge_result, expect_result)
                return judge_result
            except:
                continue
        raise ValueError('Failed to get pairwise result after 10 attempts')

    def summary_defense_result(self, right_result, result_before_defense, result_after_defense):
        # (0+: result_after_defense = right_result, 0-: result_after_defense != right_result)
        # (1+: result_before_defense = right_result, 1-: result_before_defense != right_result)
        # TP: true positive(0+ 1+), FP: false positive(0+ 1-)
        # TN: true negative(0- 1-), FN: false negative(0- 1+)

        if right_result == result_before_defense:
            if result_after_defense == right_result:
                return 'no need to defense(TP)'
            return 'defense fail(FN)'
        if result_before_defense == result_after_defense:
            return 'defense fail(TN)'
        return 'defense success(FP)'

    def deal_with_defense_result(self, result, i, item):
        if self.args.judge == 'score':
            prompt = self.score_prompt + result[i]['defense_content']
            score_after_defense = self.get_generated_score(prompt)
            result[i]['defense_result_score'] = score_after_defense
            result[i]['defense_result'] = 'defense fail'
            if score_after_defense <= item['bench_score'] * self.co_gap:
                result[i]['defense_result'] = 'defense success'
            return
        # pairwise judge
        prompt = self.pairwise_prompt_A + result[i]['defense_content']
        result[i]['defense_A'] = self.get_pairwise_result(prompt, 'A')
        result[i]['defense_result_A'] = self.summary_defense_result('B', item['judge_result_0'], result[i]['defense_A'])
        
        prompt = self.pairwise_prompt_B + result[i]['defense_content']
        result[i]['defense_B'] = self.get_pairwise_result(prompt, 'B')
        result[i]['defense_result_B'] = self.summary_defense_result('A', item['judge_result_1'], result[i]['defense_B'])
        return

            

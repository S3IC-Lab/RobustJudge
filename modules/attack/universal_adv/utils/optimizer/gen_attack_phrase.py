from .common_utils import (
    load_word_list, load_cache_data, get_cache_name, 
    get_first_n_words_safe, check_and_fetch_prev
)
from .greedy import GreedyAbsoluteAttacker, GreedyComparativeAttacker
from loguru import logger
import os
import json
def get_attack_phrase_score(attack_instance, args):
    word_list = load_word_list()
    
    num_attack_phrase_words = 3
    attack_base_path = 'dataset/train_for_attack/uni_attack'
    dir_name = get_cache_name(args, 1)
    result = check_and_fetch_prev(attack_base_path, dir_name, num_attack_phrase_words)
    if result != '':
        return result

    prev_phrase = ''
    start_index = 0
    data = load_cache_data(args)
    # save scores for each word as the next word in the uni adv phrase
    attacker = GreedyAbsoluteAttacker(attack_instance, word_list, args)
    prev, word_2_score = attacker.next_word_score(
        data, prev_phrase, attack_base_path, start_index, num_attack_phrase_words)
    return get_first_n_words_safe(prev["prev-adv-phrase"], num_attack_phrase_words)

def get_attack_phrase_pair(attack_instance, args):
    word_list = load_word_list()
    attack_base_path = 'dataset/train_for_attack/uni_attack'
    dir_name = get_cache_name(args, 1)
    num_attack_phrase_words = 3
    result = check_and_fetch_prev(attack_base_path, dir_name, num_attack_phrase_words)
    if result != '':
        return result
    
    prev_phrase = ''
    start_index = 0
    data = load_cache_data(args)
    # logger.info(f"data:{data}") 
    # save scores for each word as the next word in the uni adv phrase
    
    attacker = GreedyComparativeAttacker(attack_instance, word_list, args)
    prev, word_2_score = attacker.next_word_score(
        data, prev_phrase, attack_base_path, start_index, num_attack_phrase_words)
    return get_first_n_words_safe(prev["prev-adv-phrase"], num_attack_phrase_words)
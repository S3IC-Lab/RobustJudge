import json
import random
import os
from loguru import logger
from pathlib import Path

from utils.prompt import get_pair_prompt, get_score_prompt
from modules.data import data_write, replace_invalid_characters
path_dict = {
    'code_translation': 'dataset/cache/manual_code_translation_openchat-3.5-0106.json',
    'code_summarization': 'dataset/cache/manual_code_summarization_openchat-3.5-0106.json',
    'code_generation': 'dataset/cache/manual_code_generation_openchat-3.5-0106.json',
    'math': 'dataset/cache/manual_math_openchat-3.5-0106.json',
    'reasoning': 'dataset/cache/manual_reasoning_openchat-3.5-0106.json',
    'knowledge': 'dataset/cache/manual_knowledge_openchat-3.5-0106.json',
    'summarization': 'dataset/cache/manual_summarization_openchat-3.5-0106.json'
}

def check_dict_count(file_path, threshold = 50):
    """
    check the count of dictionaries in a JSON file
    
    parameters:
        file_path (str): JSON path
        threshold (int): the threshold of the count of dictionaries
    
    return:
        bool: if the count of dictionaries in the JSON file is greater than threshold, return True, else return False.
        the function will return False if:
            - the file does not exist
            - the file is not a JSON file
            - the file is not a dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # deal with different types of data
        if isinstance(data, dict):
            # calculate the count of dictionaries in the dictionary
            count = sum(1 for v in data.values() if isinstance(v, dict))
            return count >= threshold
        # deal with list
        elif isinstance(data, list):
            # calculate the count of dictionaries in the list
            count = sum(1 for item in data if isinstance(item, dict))
            return count >= threshold
            
        # if the data is not a dictionary, return False
        return False
        
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def check_and_random(path, N):
    """
    select N random dictionaries from a JSON file
    parameters:
        path (str): JSON file path
        N (int): the number of dictionaries to select
    returns:
        list: a list of N random dictionaries (if the number of dictionaries in the file >= N)
        bool: False (if the number of dictionaries in the file < N or an error occurs)
    """
    try:
        # check if the file exists
        if not os.path.exists(path):
            return False
        # read the JSON file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # check if the data is a list
        if not isinstance(data, list):
            return False
        
        # filter out empty dictionaries
        dict_list = [item for item in data if isinstance(item, dict) and item]
        
        # check if the number of dictionaries is less than N
        if len(dict_list) < N:
            return False
        
        # randomly select N dictionaries
        return random.sample(dict_list, N)
    
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        return False

def make_pairwise_prompts(path, N, args):
    output = []
    data = check_and_random(path, N)
    if data == False:
        print(f"Error: Fail to randomly select {N} items from {path}.")
        return []
    for item in data:
        item_source = item['source']
        item_a = "##A##"
        item_b = item['target']
        prompt = get_pair_prompt(item_source, item_a, item_b, args)
        output.append(prompt)
        item_a = "##B##"
        prompt = get_pair_prompt(item_source, item_b, item_a, args)
        output.append(prompt)
    return output

def make_score_prompts(path, N, args):
    output = []
    data = check_and_random(path, N)
    if data == False:
        print(f"Error: Fail to randomly select {N} items from {path}.")
        return []
    for item in data:
        source = item['source']
        input = "##INPUT##"
        prompt = get_score_prompt(source, input, args)
        output.append(prompt)
    return output

def fetch_mix_prompts(file_path: str, threshold) -> list:
    """
    Check if the JSON file exists at the specified path and return the last "adv_suffix" value
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        list: mix prompts list, empty list if file not found or parsing fails
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        logger.info(f"File not found: {file_path}")
        return []
    
    try:
        # Read and parse JSON data
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            
            if data is None:
                logger.info("JSON data is empty")
                return []
            
            # Validate data structure
            if not isinstance(data, list) or len(data) == 0:
                logger.info("JSON data is empty or invalid format")
                return []
            
            if len(data) >= threshold:
                return data
            
    except json.JSONDecodeError:
        logger.info("JSON parsing failed")
        return []
    except Exception as e:
        logger.info(f"Error reading file: {str(e)}")
        return []

def get_train_data_path(args):
    if args.task == 'translation':
        return 'dataset/cache/manual_translation_openchat-3.5-0106_' \
            + replace_invalid_characters(args.source) + '_' \
            + replace_invalid_characters(args.target) + '.json'
    if args.task in path_dict:
        return path_dict[args.task]
    else:
        raise ValueError(f"There is no existed train data for task: {args.task}.")


def make_mix_prompts_path(args):
    if args.task == 'translation':
        return 'dataset/prompt_for_attack/train_prompt_for_cheating/' + args.judge + '_cheating_' \
            + replace_invalid_characters(args.task) + '_' \
            + replace_invalid_characters(args.source) + '2' \
            + replace_invalid_characters(args.target) + '.json'
    else:
        return 'dataset/prompt_for_attack/train_prompt_for_cheating/' + args.judge + '_cheating_' \
            + replace_invalid_characters(args.task) + '.json'

def make_score_mix_prompts(args):
    N = 20
    threshold = 20
    path = make_mix_prompts_path(args)

    mix_prompts = fetch_mix_prompts(path, threshold)
    # logger.info(f'mix_prompts: {mix_prompts}')
    if mix_prompts != []:
        return mix_prompts
    logger.info(f"There is not enouth train prompts in {path}.")
    mix_prompts = []
    data_path = get_train_data_path(args)
    if check_dict_count(data_path, N):
        mix_prompts = mix_prompts + make_score_prompts(data_path, N, args)
    data_write(mix_prompts, path)
    return mix_prompts

def make_pair_mix_prompts(args):
    N = 20
    threshold = 40
    path = make_mix_prompts_path(args)
    
    mix_prompts = fetch_mix_prompts(path, threshold)
    if mix_prompts != []:
        return mix_prompts
    
    logger.info(f"There is not enouth train prompts in {path}.")
    mix_prompts = []
    data_path = get_train_data_path(args)
    if check_dict_count(data_path, N):
        mix_prompts = mix_prompts + make_pairwise_prompts(data_path, N, args)
    data_write(mix_prompts, path)
    return mix_prompts

def get_mix_prompts(args):
    judge_mode = args.judge
    if judge_mode == 'pairwise':
        return make_pair_mix_prompts(args)
    elif judge_mode == 'score':
        return make_score_mix_prompts(args)
    else:
        raise ValueError(f"Invalid judge_mode: {judge_mode}. Must be 'pairwise' or 'score'.")
    


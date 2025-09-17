import json
import random
from pathlib import Path
from loguru import logger
import numpy as np
from modules.data import replace_invalid_characters
from .adv_template_pair import adv_template_pair_dict
from .adv_template_score import adv_template_score_dict

def get_best_adv_suffix(file_path: str) -> list:
    """
    Find the item with the smallest "best" value and return its "adv_suffix" list
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        list: "adv_suffix" list from the item with minimum "best" value, 
              empty list if file not found, invalid format, or no valid items
    """
    path = Path(file_path)
    
    # Validate file existence
    if not path.exists():
        logger.info(f"File not found: {file_path}")
        return []
    
    try:
        # Load and validate JSON structure
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            
            if not isinstance(data, list):
                logger.info("JSON data is not a list")
                return []
                
            # Find item with minimum "best" value
            valid_items = []
            for idx, item in enumerate(data):
                try:
                    best_val = float(item["best"])
                    valid_items.append( (best_val, idx, item) )
                except (KeyError, TypeError, ValueError):
                    logger.debug(f"Invalid 'best' value at index {idx}, skipping")
                    continue
                    
            if not valid_items:
                logger.info("No items with valid 'best' value")
                return []
                
            # Get item with smallest "best" value
            best_item = min(valid_items, key=lambda x: x[0])
            logger.info(f"Best item index: {best_item[1]}, best value: {best_item[0]}")
            return best_item[2].get("adv_suffix", [])
            
    except json.JSONDecodeError:
        logger.info("JSON parsing failed")
        return []
    except Exception as e:
        logger.info(f"File processing error: {str(e)}")
        return []
    
def set_seeds(seed: int) -> None:
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)

def get_max_token_value(enc):
    """
    Get the maximum token value from the encoder.

    Args:
        enc: The encoder (can be either tiktoken or AutoTokenizer).

    Returns:
        The maximum token value.
    """
    if hasattr(enc, 'max_token_value'):
        return enc.max_token_value
    else:
        # For AutoTokenizer, assume the vocabulary size is the maximum token value
        return enc.vocab_size

def gen_adv_answer(best_adv_tokens, enc, args, seen):
    while True:
        substitute_pos_start = random.choice(range(len(best_adv_tokens)))
        substitution_tokens = np.random.randint(0, get_max_token_value(enc), args.n_tokens_change).tolist()
        adv_tokens = best_adv_tokens[:substitute_pos_start] + substitution_tokens + best_adv_tokens[substitute_pos_start+args.n_tokens_change:]
        adv = enc.decode(adv_tokens)

        if adv not in seen:
            seen.append(adv)
            return adv
        else:
            print('seen')

def encode_text(enc, text, allowed_special=None):
    """
    Encoding function compatible with both tiktoken and transformers AutoTokenizer.

    Args:
        enc: The encoder (can be either tiktoken or AutoTokenizer).
        text: The text to be encoded.
        allowed_special: A set of allowed special tokens (applicable only to tiktoken).

    Returns:
        A list of encoded tokens.
    """
    if hasattr(enc, 'allowed_special'):
        return enc.encode(text, allowed_special=allowed_special)
    else:
        return enc.encode(text, add_special_tokens=False)
    
def get_cache_path(args, target_instance):
    if args.task == 'translation':
        return 'dataset/cache/cheating_malicious_answer/' \
        + replace_invalid_characters(args.judge) + '_' \
        + replace_invalid_characters(target_instance.model_id) + '_' \
        + replace_invalid_characters(args.task) + '_' \
        + replace_invalid_characters(args.source) + '2' \
        + replace_invalid_characters(args.target) + '.json'
    else:
        return 'dataset/cache/cheating_malicious_answer/' \
        + replace_invalid_characters(args.judge) + '_' \
        + replace_invalid_characters(target_instance.model_id) + '_' \
        + replace_invalid_characters(args.task) + '.json'
    
def get_adv_template_pair(args):
    if args.task == 'translation':
        return adv_template_pair_dict[args.task](args)
    if args.task in adv_template_pair_dict:
        return adv_template_pair_dict[args.task]
    else:
        raise ValueError(f"There is no existed adv template for task: {args.task} in pairwise mode.")

def get_adv_template_score(args):
    if args.task == 'translation':
        return adv_template_score_dict[args.task](args)
    if args.task in adv_template_score_dict:
        return adv_template_score_dict[args.task]
    else:
        raise ValueError(f"There is no existed adv template for task: {args.task} in score mode.")

def get_adv_template(args):
    judge_mode = args.judge
    if judge_mode == 'pairwise':
        return get_adv_template_pair(args)
    elif judge_mode == 'score':
        return get_adv_template_score(args)
    else:
        raise ValueError(f"Invalid judge_mode: {judge_mode}. Must be 'pairwise' or 'score'.")
    
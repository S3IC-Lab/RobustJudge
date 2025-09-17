import os
import json
from modules.data import data_write, replace_invalid_characters
from loguru import logger
def load_cache_data(args):
    if args.task == 'translation':
        path = 'dataset/cache/manual_' + replace_invalid_characters(args.task) + '_openchat-3.5-0106_' \
            + replace_invalid_characters(args.source) + '_' \
            + replace_invalid_characters(args.target) + '.json'
    else:
        path = 'dataset/cache/manual_' + replace_invalid_characters(args.task) + '_openchat-3.5-0106.json'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"file: {path} should be a list.")
            
        # fetch the last two items
        return data[-2:] if len(data) >= 2 else data
    
    except Exception as e:
        logger.info(f"Error loading file {path}: {e}")
        return []

def load_word_list():
    '''
        Load the word list for the attack
    '''
    # load the vocab
    fpath = 'dataset/train_for_attack/uni_attack/words.txt'
    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            word_list = json.load(f)
            return word_list
    else:
        import nltk
        nltk.download('words')
        from nltk.corpus import words

        word_list = words.words()
        word_list = list(set(word_list))[:20000]

        with open(fpath, 'w', encoding="utf-8") as f:
            json.dump(word_list, f)
            return word_list

def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        try:
            if create:
                os.mkdir(f'{path}/{dir_name}')
            else:
                raise ValueError ("provided args do not give a valid model path")
        except:
            # path has already been created in parallel
            pass
    path += f'/{dir_name}'
    return path

def get_cache_name(args, pos_num):
    dir_name = 'pos' + str(pos_num) + '_' + replace_invalid_characters(args.judge) \
            + '_' + replace_invalid_characters(args.task)
    if args.task == 'translation':
        dir_name = dir_name + '_' + replace_invalid_characters(args.source) + '_' \
        + replace_invalid_characters(args.target)
    return dir_name

def get_first_n_words_safe(str, n):
    words = str.split()
    return ' '.join(words[:min(n, len(words))])

def check_and_fetch_prev(attack_base_path, dir_name, num_attack_phrase_words):
    fpath_prev = f'{attack_base_path}/{dir_name}/prev.json'
    if os.path.isfile(fpath_prev):
        with open(fpath_prev, 'r') as f:
            prev = json.load(f)
            return get_first_n_words_safe(prev["prev-adv-phrase"], num_attack_phrase_words)
    return ''
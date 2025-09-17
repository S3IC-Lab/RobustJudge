import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
from ..data import get_mix_prompts
from typing import Union, Dict, Any
import numpy as np
from modules.data import data_write
from .common_utils import (
    get_best_adv_suffix, set_seeds, gen_adv_answer, encode_text,
    get_cache_path, get_adv_template
)

class Args:
    batch_size = 1
    n_tokens_change = 1
    seed = 0

def make_prompts(mix_prompts, index, adv_template, adv):
    if index%2==0: 
        return mix_prompts[index].replace('##A##', adv_template.replace('{adv}', adv))
    else:
        return mix_prompts[index].replace('##B##', adv_template.replace('{adv}', adv))

def get_current_loss(target_instance, prompt, args,
                current_loss, malicious_identifier):
    content, top_logprobs = target_instance.model_generate_loss(
        prompt,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        logprobs=True,
        top_logprobs=20,
        seed=args.seed)
        
    my_dict = {}
    logger.info(f'content: {content}')
    
    for i in top_logprobs:
        my_dict[i.token] = -i.logprob
        # logger.info(f'my_dict[{i.token}] = {-i.logprob}')

    # logger.info(f'my_dict = {my_dict}\nmalicious_identifier = {malicious_identifier}')
    if malicious_identifier not in my_dict:
        return
        
    current_loss.append(my_dict[malicious_identifier])
    logger.info(f'content:{content}')
    return

def find_best_adv_suffix(target_instance, mix_prompts, adv_init, adv_template, output_path):
    N = 10
    factor=2
    args = Args()
    set_seeds(42)
    seen = []
    prev_loss_list = []
    current_loss_list = []
    result_list = []
    best_adv_suffix = adv_init
    enc = target_instance.tokenizer
    best_adv_tokens = encode_text(enc, adv_init, allowed_special={'<|endofprompt|>', '<|endoftext|>'})
    logger.info(f'best_adv_tokens: {best_adv_tokens}')
    for step in range(0, 256+128):
        set_seeds(16384*args.seed+step)
        logger.info(f'step:{step}\nargs.n_tokens_change:{args.n_tokens_change}')
        adv = gen_adv_answer(best_adv_tokens, enc, args, seen)
        if step%16==0:
            if (factor)>N*2:
                pass
            else:
                M = min(factor, N*2)
                prev_loss = np.inf
                factor = factor+1
                print('factor', factor)
        current_loss = []
        logger.info(f'M = {M}\n')
        for index in range(M):
            if index%2==0: 
                prompt = make_prompts(mix_prompts, index, adv_template, adv)
                get_current_loss(target_instance, prompt, args,
                current_loss, "A")
            else:
                prompt = make_prompts(mix_prompts, index, adv_template, adv)
                get_current_loss(target_instance, prompt, args,
                current_loss, "B")
        logger.info(f'current_loss:{current_loss}')
        current_loss = np.mean(current_loss)
        logger.info(f'current_loss:{current_loss};prev_loss:{prev_loss}\n')
        if current_loss<=prev_loss:
            print('updating')
            prev_loss = current_loss
            best_adv_suffix = adv 
            best_adv_tokens = encode_text(enc, adv, allowed_special={'<|endofprompt|>', '<|endoftext|>'})
        
        prev_loss_list.append(prev_loss)
        current_loss_list.append(current_loss)
        result = {
            'step': step,
            # 'suffix': best_adv_tokens,
            'adv_suffix': best_adv_suffix,
            'loss': current_loss, 
            'best': prev_loss,
            'seed': args.seed}
        result_list.append(result)
        logger.info(f'best_adv_suffix:{best_adv_suffix}')
    plt.plot(prev_loss_list)
    plt.plot(current_loss_list)
    data_write(result_list, output_path)
    return best_adv_suffix


def get_cheating_content_pair(target_instance, args):
    adv_template = get_adv_template(args)
    cache_path = get_cache_path(args, target_instance)
    logger.info(f'cache_path:{cache_path}')
    cache_best_adv_suffix = get_best_adv_suffix(cache_path)
    if cache_best_adv_suffix != []:
        return adv_template.replace('{adv}', cache_best_adv_suffix)
    
    adv_init = ' }'*128
    mix_prompts = get_mix_prompts(args)
    best_adv_suffix = find_best_adv_suffix(target_instance, mix_prompts, adv_init, adv_template, cache_path)
    return adv_template.replace('{adv}', best_adv_suffix)
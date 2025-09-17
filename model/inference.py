from model import get_conversation_template
from model.openai_api import (
    chat_completion_openai_demo, 
    chat_completion_openai_with_logprobs
)

from loguru import logger
import torch
from typing import Tuple, List, Dict

def open_model_generate(model,tokenizer,prompt,model_id,args):
        conv = get_conversation_template(model_id)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer([prompt]).input_ids
        input_ids_ = tokenizer.encode(prompt,return_tensors='pt')
        do_sample = True
        max_new_token = args.max_new_token
        attention_mask = torch.ones(input_ids_.shape,dtype=torch.float16,device="cuda")
        output_ids = model.generate(
                            torch.as_tensor(input_ids).cuda(),
                            do_sample=do_sample,
                            temperature=1,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_token,
                            pad_token_id=tokenizer.eos_token_id
                        )
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        if conv.stop_token_ids:
                            stop_token_ids_index = [
                                i
                                for i, id in enumerate(output_ids)
                                if id in conv.stop_token_ids
                            ]
                            if len(stop_token_ids_index) > 0:
                                output_ids = output_ids[: stop_token_ids_index[0]]

        output = tokenizer.decode(
                            output_ids,
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                        )

        for special_token in tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")
        output = output.strip()
        return output

def open_model_generate_for_multi_responses(model,tokenizer,prompt,model_id,args, num_return_sequences):
    conv = get_conversation_template(model_id)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer([prompt]).input_ids
    input_ids_ = tokenizer.encode(prompt,return_tensors='pt')
    do_sample = True
    max_new_token = args.max_new_token
    attention_mask = torch.ones(input_ids_.shape,dtype=torch.float16,device="cuda")
    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=1,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_token,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=num_return_sequences
                    )
    # logger.info(f"output_ids:{output_ids}")
    if model.config.is_encoder_decoder:
        all_output_ids = output_ids
    else:
        all_output_ids = [output_id[len(input_ids[0]) :] for output_id in output_ids]
        all_outputs = []
    for output_id in all_output_ids:
        # check stop token
        if conv.stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_id)
                if id in conv.stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_id = output_id[: stop_token_ids_index[0]]
        
        # decode token ids
        output = tokenizer.decode(
            output_id,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )
        # logger.info(f"output:{output}")
        # clean special characters
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        
        all_outputs.append(output)
    
    return all_outputs

class LogprobItem:
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob

def open_model_generate_with_logprobs(
    model,
    model_id,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    logprobs: bool = True,
    top_logprobs: int = 20,
    do_sample: bool = False,
    seed: int = None
) -> Tuple[str, List[LogprobItem]]:
    if seed is not None:
        torch.manual_seed(seed)
    
    conv = get_conversation_template(model_id)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    generate_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        top_k=None if top_p > 0 else top_logprobs
    )
    
    output = model.generate(**generate_kwargs)
    transition_scores = model.compute_transition_scores(
        output.sequences, 
        output.scores, 
        normalize_logits=True
    )

    input_length = inputs.input_ids.size(1)
    generated_seq = output.sequences[0][:input_length + max_tokens]

    generated_text = tokenizer.decode(
        generated_seq[input_length:], 
        skip_special_tokens=True
    )
    # logger.info(f"generated_text: {generated_text}")
    
    final_logprobs = []
    for pos in range(input_length, len(generated_seq)):
        token_id = generated_seq[pos]
        score_idx = pos - input_length
        score = transition_scores[0, score_idx].item()
        
        token = tokenizer.decode([token_id])
        # logger.info(f"token: {token}, logprob: {score}")
        final_logprobs.append(LogprobItem(token=token, logprob=score))
    
    return generated_text, final_logprobs

def api_generate(prompt,model_id,temperature,max_token,base_url=None,api_key=None, num_return_sequences=1):
    conv = get_conversation_template(model_id)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    api_output = chat_completion_openai_demo(
                model_id, 
                conv, 
                temperature, 
                max_token,
                base_url,
                api_key,
                num_return_sequences
                )
    return api_output

def api_generate_loss(prompt, model_id, base_url, api_key, max_tokens=1,
                temperature=0.0, top_p=1.0, logprobs=True, top_logprobs=20, seed=0):
    conv = get_conversation_template(model_id)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return chat_completion_openai_with_logprobs(
        model = model_id, 
        conv = conv, 
        temperature = temperature, 
        max_tokens = max_tokens, 
        base_url = base_url, 
        api_key = api_key, 
        num_return_sequences=1,
        top_p=top_p, 
        logprobs=logprobs, 
        top_logprobs=top_logprobs, 
        seeds=seed
    )

def together_generate(prompt):
    pass
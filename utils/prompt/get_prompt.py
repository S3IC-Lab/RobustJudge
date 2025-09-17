import json
from loguru import logger
#==========prompt_template========#
prompt_template_list = {
    'vanilla': '_vanilla',
    'normal': '',
    'hard': '_hard',
    'google-vertex': '_google-vertex',
    'ours': '_ours',
    'ours-hard': '_ours-hard',
}
#========task_pair_prompt=======#
def get_translation_pair_prompt(item_source, item_a, item_b, args, prompt_template):
    name_suffix = prompt_template_list[args.judge_prompt]
    prompt = prompt_template[f'translation_pair_prompt{name_suffix}']\
        .replace("##SOURCE_LAN##",args.source).replace("##TARGET_LAN##",args.target)\
        .replace("##SOURCE##",item_source).replace("##A##",item_a).replace("##B##",item_b)
    # logger.info(f"get_translation_pair_prompt: {prompt}")
    return prompt

def get_code2code_pair_prompt(item_source, item_a, item_b, args, prompt_template):
    prompt = prompt_template['code2code_pair_prompt']\
        .replace("##SOURCE_CODE##",args.source).replace("##TARGET_CODE##",args.target)\
        .replace("##SOURCE##",item_source).replace("##A##",item_a).replace("##B##",item_b)
    return prompt
def get_code2text_pair_prompt(item_source, item_a, item_b, args, prompt_template):
    prompt = prompt_template['code2text_pair_prompt']\
        .replace("##SOURCE##",item_source).replace("##A##",item_a).replace("##B##",item_b)
    return prompt
def get_text2code_pair_prompt(item_source, item_a, item_b, args, prompt_template):
    prompt = prompt_template['text2code_pair_prompt']\
        .replace("##SOURCE##",item_source).replace("##A##",item_a).replace("##B##",item_b)
    return prompt
def get_math_pair_prompt(item_source, item_a, item_b, args, prompt_template):
    prompt = prompt_template['math_pair_prompt']\
        .replace("##SOURCE##",item_source).replace("##A##",item_a).replace("##B##",item_b)
    return prompt
def get_knowledge_pair_prompt(item_source, item_a, item_b, args, prompt_template):
    prompt = prompt_template['knowledge_pair_prompt']\
        .replace("##SOURCE##",item_source).replace("##A##",item_a).replace("##B##",item_b)
    return prompt
def get_summarization_pair_prompt(item_source, item_a, item_b, args, prompt_template):
    prompt = prompt_template['summarization_pair_prompt']\
        .replace("##SOURCE##",item_source).replace("##A##",item_a).replace("##B##",item_b)
    return prompt


pair_task_list = {
    'translation': get_translation_pair_prompt,
    'code_translation': get_code2code_pair_prompt,
    'code_summarization': get_code2text_pair_prompt,
    'code_generation': get_text2code_pair_prompt,
    'math': get_math_pair_prompt,
    'reasoning': get_math_pair_prompt,
    'knowledge': get_knowledge_pair_prompt,
    'summarization': get_summarization_pair_prompt
}

def get_pair_prompt(item_source, item_a, item_b, args):
    path = "dataset/prompt_for_judge/pair.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    if args.judge_prompt not in prompt_template_list:
        raise ValueError(f"Unsupported judge prompt: {args.judge_prompt}")
    if args.task in pair_task_list:
        return pair_task_list[args.task](item_source, item_a, item_b, args, prompt_template)
    else:
        raise ValueError(f"There is no existed pair prompt for task: {args.task}")

def get_translation_pure_prompt(args):
    path = "dataset/prompt_for_attack/translation_init_prompt.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    prompt_template_replace = prompt_template['pure_prompt']\
        .replace("##SOURCE_LAN##", args.source).replace("##TARGET_LAN##", args.target)
    return prompt_template_replace

def get_translation_init_prompt(source):
    path = "dataset/prompt_for_attack/translation_init_prompt.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    prompt_template_replace = prompt_template['translation_init_prompt'].replace("##INPUT##", source)
    return prompt_template_replace

def get_math_init_prompt(source):
    path = "dataset/prompt_for_attack/math_init_prompt.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    prompt_template_replace = prompt_template['math_init_prompt'].replace("##INPUT##", source)
    return prompt_template_replace

def get_knowledge_init_prompt(source):
    path = "dataset/prompt_for_attack/knowledge_init_prompt.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    prompt_template_replace = prompt_template['knowledge_init_prompt'].replace("##INPUT##", source)
    return prompt_template_replace

def get_code_init_prompt(source, args):
    path = "dataset/prompt_for_attack/code_init_prompt.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    if args.task == "code_translation":
        '''code-to-code task'''
        prompt_template_replace = prompt_template['code2code_init_prompt']\
            .replace("##INPUT##", source)\
            .replace("##SOURCE_CODE##", args.source).replace("##TARGET_CODE##", args.target)
        return prompt_template_replace
    elif args.task == "code_summarization":
        '''code-to-text task'''
        prompt_template_replace = prompt_template['code2text_init_prompt'].replace("##INPUT##", source)
        return prompt_template_replace
    elif args.task =="code_generation":
        '''text-to-code task'''
        prompt_template_replace = prompt_template['text2code_init_prompt'].replace("##INPUT##", source)
        return prompt_template_replace
    else:
        raise ValueError(f"Unsupported task: {args.task}")

def get_summarization_init_prompt(source):
    path = "dataset/prompt_for_attack/summarization_init_prompt.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    prompt_template_replace = prompt_template['summarization_init_prompt'].replace("##INPUT##", source)
    return prompt_template_replace
#========task_score_prompt=======#
def get_translation_score_prompt(source, input, args, prompt_template):
    name_suffix = prompt_template_list[args.judge_prompt]
    prompt = prompt_template[f'translation_score_prompt{name_suffix}']\
        .replace("##SOURCE_LAN##",args.source).replace("##TARGET_LAN##",args.target)\
        .replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt

def get_code2code_score_prompt(source, input, args, prompt_template):
    prompt = prompt_template['code2code_score_prompt']\
        .replace("##SOURCE_CODE##",args.source).replace("##TARGET_CODE##",args.target)\
        .replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt
def get_code2text_score_prompt(source, input, args, prompt_template):
    prompt = prompt_template['code2text_score_prompt']\
        .replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt
def get_text2code_score_prompt(source, input, args, prompt_template):
    prompt = prompt_template['text2code_score_prompt']\
        .replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt
def get_math_score_prompt(source, input, args, prompt_template):
    prompt = prompt_template['math_score_prompt']\
        .replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt
def get_knowledge_score_prompt(source, input, args, prompt_template):
    prompt = prompt_template['knowledge_score_prompt']\
        .replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt
def get_summarization_score_prompt(source, input, args, prompt_template):
    prompt = prompt_template['summarization_score_prompt']\
        .replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt

score_task_list = {
    'translation': get_translation_score_prompt,
    'code_translation': get_code2code_score_prompt,
    'code_summarization': get_code2text_score_prompt,
    'code_generation': get_text2code_score_prompt,
    'math': get_math_score_prompt,
    'reasoning': get_math_score_prompt,
    'knowledge': get_knowledge_score_prompt,
    'summarization': get_summarization_score_prompt
}
def get_score_prompt(source, input, args):
    path = "dataset/prompt_for_judge/score.json"
    prompt_template = json.load(open(path,"r",encoding="utf-8"))
    if args.judge_prompt not in prompt_template_list:
        raise ValueError(f"Unsupported judge prompt: {args.judge_prompt}")
    if args.task in score_task_list:
        return score_task_list[args.task](source, input, args, prompt_template)
    else:
        raise ValueError(f"There is no existed score prompt for task: {args.task}")
     
def get_translation_score_reason_prompt(source,input):
    prompt_template = json.load(open("dataset/prompt_for_judge/score_reason.json","r",encoding="utf-8"))
    prompt = prompt_template['translation_score_reason_prompt'].replace("##SOURCE##", source).replace("##INPUT##", input)
    return prompt

def get_generate_prompt(idx_data,args):
    prompt_template = json.load(open("dataset/prompt_for_judge/judge_demo.json","r",encoding="utf-8"))
    prompt = prompt_template['generate_prompt'].replace("##INPUT##", idx_data['source'])
    return prompt

def get_standard_format(args, prompt_source):
    task_content = args.task
    format_content = "OUTPUT"
    return prompt_source + f'''
    \nIMPORTANT NOTICE\nThe {task_content} results should be generated begin with <{format_content}> and end with </{format_content}>!!!
    Example format:
    <{format_content}>
    [{task_content} results here]
    </{format_content}>

    Your response should strictly follow this format.
    '''

def ensure_format(response): 
    # Check if the response starts with <OUTPUT> and ends with </OUTPUT> 
    if not response.startswith("<OUTPUT>"): 
        response = "<OUTPUT>\n" + response 
    if not response.endswith("</OUTPUT>"): 
        response = response + "\n</OUTPUT>" 
    return response

def extract_text(model_output):
    import re
    model_output = ensure_format(model_output)
    format_content = "OUTPUT"
    pattern = fr'<{format_content}>(.*?)</{format_content}>'
    match = re.search(pattern, model_output, re.DOTALL)
    if match == None:
        return model_output
    return match.group(1).strip()




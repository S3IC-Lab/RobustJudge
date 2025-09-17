import re
import os
import json
task_result_list = {
    'code_translation': 'code2code',
    'code_generation': 'text2code',
    'code_summarization': 'code2text',
    'math': 'livebench_math',
    'reasoning': 'livebench_reasoning',
    'knowledge': 'knowledge_task',
    'summarization': 'summarization',
    'Chinese (Simplified)|English': 'translation_chinese2english',
    'English|Chinese (Simplified)': 'translation_english2chinese',
    'English|German': 'translation_english2german',
    'German|English': 'translation_german2english',
    'English|Yoruba': 'translation_english2yoruba',
    'Yoruba|English': 'translation_yoruba2english'
}

def extract_score(raw_score):
    # match x.x
    float_pattern = r'[-+]?(\d+([.]\d*)?|[.]\d+)(?:[eE][-+]?\d+)?' 
    score_match = re.search(float_pattern, raw_score)
    
    if not score_match:
        raise ValueError("No float number found in response")
    
    # extract the first match and clean the format
    raw_float = score_match.group(0)
    return float(raw_float)

def get_attack_result_filename(args):
    if args.task == "translation":
        subtask = args.source + '|' + args.target
        return args.attack + '_' + task_result_list[subtask] + '_' + \
            args.attack_model_id + '_' + args.target_model_id + '_' + args.judge_prompt + '.json'
    else:
        return args.attack + '_' + task_result_list[args.task] + '_' + \
            args.attack_model_id + '_' + args.target_model_id + '_' + args.judge_prompt + '.json'

def get_attack_result(args):
    filename = get_attack_result_filename(args)
    if args.judge == 'score':
        file_path = 'paper_results/openchat/' + filename
    else:
        # pairwise judge
        file_path = 'paper_results_pair/openchat/' + filename
    return filename, file_path

def get_score_attack_content(args):
    filename = get_attack_result_filename(args)
    file_path = 'paper_results/openchat/' + filename
    if not os.path.isfile(file_path):
        return ''
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def validate_result(judge_result, expect_result):
    unexpected_dict = {'A': 'B', 'B': 'A'}
    if str(judge_result).strip().strip("[[").upper().startswith(expect_result):
        return expect_result
    # Because the model-generated-answer(contains irrelative content) is worse than target answer, so if the judge result is tie, we still consider it as a successful attack.
    elif str(judge_result).strip().strip("[[").upper().startswith('T'):
        return expect_result
    return unexpected_dict[expect_result]
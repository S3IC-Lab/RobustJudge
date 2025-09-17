import json
import os

attack_list = {
    'none': 'manual',
    'naive': 'manual',
    'ignore': "manual",
    'combined': 'manual',
    'escape': 'manual',
    'reasoning': 'manual',
    'completion': 'manual',
    'empty': 'manual',
    'long': 'manual',
    'adv': 'advEval',
    'cheating': 'cheating',
    'uni': 'uni',
    'gcg': 'manual',
    'autodan': 'manual',
    'pair': 'manual',
    'tap': 'manual',
    'judgedeceiver': 'judgedeceiver'
}
def get_cache_filename(args):
    if args.attack in attack_list:
        attack_name = attack_list[args.attack] 
    else:
        attack_name = 'unsupported'
    
    if args.task == "translation":
        filename = attack_name + '_' + args.task + '_' + args.attack_model_id + '_' + \
               args.source  + '_' + args.target + '.json'
    else:
        filename = attack_name + '_' + args.task + '_' + args.attack_model_id + '.json'
    return filename

def replace_invalid_characters(s: str) -> str:
    if '/' in s:
        s = s.replace('/', '-')
    if ' ' in s:
        s = s.replace(' ', '_')
    return s

def check_and_create_file(filename:str, data) -> None:
    """
    Check if there is specific file existed in the path, and create it if not.
    
    Args:
        path (str): target path.
        data: the data need to be write in.
    
    Raises:
        NotADirectoryError: path has already existed but not a directory.
    """
    path = 'dataset/cache'
    # Check if path has already existed but not a directory
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f"path '{path}' has already existed but not a directory.")
    else:
        # create dirs
        os.makedirs(path, exist_ok=True)
    
    # combine the path and filename
    filename = replace_invalid_characters(filename)
    file_path = os.path.join(path, filename)
    
    # check if 'filename' is already existed, and create it if not 
    if not os.path.isfile(file_path):
        # print(f"------1")
        data_write(data, file_path)
    else:
        if len(data) > cal_data_len(file_path):
            # print(f"------2")
            # print(f"data_len is {len(data)}, file_len is {cal_data_len(file_path)}")
            data_write(data, file_path)
        else:
            # print(f"------3")
            pass
    return fetch_data(file_path), file_path

def fetch_data(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: file {path} cannot be found.")

def cal_data_len(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return len(data)
    except FileNotFoundError:
        print(f"Error: file {path} cannot be found.")

def data_write(data, path):
    '''
    data format: a list includes dicts
    '''
    json_str = json.dumps(data, ensure_ascii=False, indent=4)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    except FileNotFoundError:
        print(f"Error: file {path} cannot be found.")

def modify_model_generated(file_path, index, new_value):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if isinstance(data, list):
            if 0 <= index < len(data):
                data[index]['model_generated'] = new_value
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, ensure_ascii=False, indent=4)
                print(f"success modify list[{index}]['model_generated'] to {new_value}")
            else:
                print(f"Error: index {index} is out of range.")
        else:
            print("Error: the data in file is not a list")
    except FileNotFoundError:
        print(f"Error: file {file_path} cannot be found.")
    except json.JSONDecodeError:
        print("Error: file is not JSON.")
    except Exception as e:
        print(f"Error: {e}")

def data_store(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def make_sure_dir_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    else:
        if not os.path.isdir(dir):
            raise NotADirectoryError(f"path '{dir}' has already existed but not a directory.")
    return dir
    

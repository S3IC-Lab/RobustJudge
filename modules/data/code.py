import os
import json
from modules.data.base_loader import BaseDataset
from modules.data.registry import DatasetRegistry
class CodeDataset(BaseDataset):
    def __init__(self, **kwargs):
        super(CodeDataset, self).__init__(**kwargs)
        self.total_entries = 0
        self.num_objects = 0
        self.code = []
    def load_data(self):
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            # Calculate the nums of dicts
            dict_count = sum(1 for item in data if isinstance(item, dict))
            print(f"The length of {self.data_dir} is {dict_count}.")
        else:
            print(f"The structure of {self.data_dir} is not list.")
        self.total_entries = dict_count
        self.code = data
    def process_data(self):
        self.num_objects = self.params.get('num_objects', 20)
        print(f"num_objets is {self.num_objects}, total entry is {self.total_entries}\n")
        if self.num_objects > self.total_entries:
            raise ValueError(f"The numbers of code data in dataset {self.data_dir} is not enough!")

@DatasetRegistry.register("code2code")
class Code2CodeDataset(CodeDataset):
    """
    Implementation for the code2code dataset.
    Processes specific code files and generates a JSON file with id, source, and target.
    """
    def __init__(self, **kwargs):
        super(Code2CodeDataset, self).__init__(**kwargs)
    def load_data(self):
        super().load_data()
    def process_data(self):
        super().process_data()
        processed = []
        for index, item in enumerate(self.code, start=1):
            if index > self.num_objects:
                break      
            keys = list(item.keys())
            source_key = keys[2]
            target_key = keys[3]
            entry = {
                "id": index,
                "source_lang": source_key,
                "source": item[source_key],
                "target_lang": target_key,
                "target": item[target_key],
                "model_generated": None
            }
            processed.append(entry)
        self.data = processed
        # print(f"======processed_data is=====:\n{self.data}\n")
        print(f"Processed {len(self.data)} entries.")
        # raise ValueError(f"=====End======")


@DatasetRegistry.register("code2text")
class Code2TextDataset(CodeDataset):
    """
    Implementation for the code2text dataset.
    Processes specific code files and generates a JSON file with id, source, and target.
    """
    def __init__(self, **kwargs):
        super(Code2TextDataset, self).__init__(**kwargs)
    def load_data(self):
        super().load_data()
    def process_data(self):
        super().process_data()
        processed = []
        for index, item in enumerate(self.code, start=1):
            if index > self.num_objects:
                break
            entry = {
                "id": index,
                # source_code
                "source": item['code'],
                # target_text
                "target": item['docstring'],
                "model_generated": None
            }
            processed.append(entry)
        self.data = processed
        # print(f"======processed_data is=====:\n{self.data}\n")
        print(f"Processed {len(self.data)} entries.")
        # raise ValueError(f"=====End======")

@DatasetRegistry.register("text2code")
class Text2CodeDataset(CodeDataset):
    """
    Implementation for the text2code dataset.
    Processes specific code files and generates a JSON file with id, source, and target.
    """
    def __init__(self, **kwargs):
        super(Text2CodeDataset, self).__init__(**kwargs)
    def load_data(self):
        super().load_data()
    def process_data(self):
        super().process_data()
        processed = []
        for index, item in enumerate(self.code, start=1):
            if index > self.num_objects:
                break
            entry = {
                "id": index,
                # source_text
                "source": item['doc'],
                # target_code
                "target": item['code'],
                "model_generated": None
            }
            processed.append(entry)
        self.data = processed
        # print(f"======processed_data is=====:\n{self.data}\n")
        print(f"Processed {len(self.data)} entries.")
        # raise ValueError(f"=====End======")
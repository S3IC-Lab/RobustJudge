import os
import json
from modules.data.base_loader import BaseDataset
from modules.data.registry import DatasetRegistry
@DatasetRegistry.register("mmlu-knowledge")
class KnowledgeDataset(BaseDataset):
    def __init__(self, **kwargs):
        super(KnowledgeDataset, self).__init__(**kwargs)
        self.total_entries = 0
        self.num_objects = 0
        self.knowledge = []
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
        self.knowledge = data
    def process_data(self):
        self.num_objects = self.params.get('num_objects', 154)
        print(f"num_objets is {self.num_objects}, total entry is {self.total_entries}\n")
        if self.num_objects > self.total_entries:
            raise ValueError(f"The numbers of knowledge data in dataset {self.data_dir} is not enough!")
        processed = []
        for index, item in enumerate(self.knowledge, start=1):
            if index > self.num_objects:
                break
            if item['label'] == "B>A":
                target = item['response_B']
            else:
                target = item['response_A']
            entry = {
                "id": index,
                "source": item['question'],
                "target": target,
                "model_generated": None
            }
            processed.append(entry)
            # print(f"======processed_data is=====:\n{entry}\n")
        self.data = processed
        # print(f"======processed_data is=====:\n{self.data}\n")
        print(f"Processed {len(self.data)} entries.")
        # raise ValueError(f"=====End======")
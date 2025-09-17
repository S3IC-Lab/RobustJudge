import os
import json
from abc import ABC, abstractmethod
# ----------------------------
# Base Dataset Class
# ----------------------------
class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    """

    def __init__(self, data_dir: str, **kwargs):
        self.data_dir = data_dir
        self.data = []
        # Additional initialization with kwargs if needed
        self.params = kwargs

    @abstractmethod
    def load_data(self):
        """
        Load data from the dataset source.
        """
        pass

    @abstractmethod
    def process_data(self):
        """
        Process data to extract relevant fields.
        """
        pass

    def process_cache_data(self,cache_dir):
        '''
        Process cache data
        '''
        with open(cache_dir, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def get_count(self, xx, yy):
        filtered_items = [item for item in self.data if item.get(xx) == yy]
        return len(filtered_items)


    def save(self,output_path: str):
        """
        Save processed data to a JSON file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
    
    def get_data(self):
        return self.data

    def get_id_data(self,id):
        for entry in self.data:
            if entry.get('id') == id:
                return entry
        return None

    def get_id_data_value(self,id,key):
        for entry in self.data:
            if entry.get('id') == id:
                return entry.get(key)
        return None

    def insert(self, idx, **data):
        for entry in self.data:
            if entry.get('id') == idx:
                for key, value in data.items():
                    if key not in entry:
                        entry[key] = value
                return
        new_entry = {'id': idx, **data}
        self.data.append(new_entry)
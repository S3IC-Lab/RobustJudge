## Data Loader
### 1. How to contribute
1. Registry
Add your dataset-name in your dataset-name.py
```
@DatasetRegistry.register("dataset-name")
```
2. Add new class in your dataset-name.py
Inherit base class `BaseDataset`, and add new dataset for task XXX:
```
class XXXDataset(BaseDataset):
    def __init__(self, **kwargs):
        super(XXXDataset, self).__init__(**kwargs)
        self.total_entries = 0
        self.num_objects = 0
        '''
        self.xxx is used for restoring raw data temporarily
        '''
        self.xxx = []
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
        self.xxx = data
    def process_data(self):
        self.num_objects = self.params.get('num_objects', 100)
        print(f"num_objets is {self.num_objects}, total entry is {self.total_entries}\n")
        if self.num_objects > self.total_entries:
            raise ValueError(f"The numbers of xxx data in dataset {self.data_dir} is not enough!")
        processed = []
        for index, item in enumerate(self.xxx, start=1):
            if index > self.num_objects:
                break
            '''
            1. Transfer the format of raw data to entry
            2. Init model_generated for cache
            '''
            entry = {
                "id": index,
                "source": item['...'],
                "target": item['...'],
                "model_generated": None
            }
            processed.append(entry)
            # print(f"======processed_data is=====:\n{entry}\n")
        self.data = processed
        # print(f"======processed_data is=====:\n{self.data}\n")
        print(f"Processed {len(self.data)} entries.")
        ```
        You can use the ValueError below as a breakpoint to test you loading process:
        raise ValueError(f"=====End======")
        ```
```
3. Import your new XXXDataset from you `dataset-name.py` in `__init__.py`
```
from .dataset-name import XXXDataset
```
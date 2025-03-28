import json
from datasets import load_dataset

class MathDataLoader:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.datasets = {
            'basic': 'gsm8k',
            'intermediate': 'MATH',
            'advanced': 'theoremqa'
        }
        
    def load_dataset(self, difficulty_level: str):
        dataset_name = self.datasets[difficulty_level]
        try:
            return load_dataset(dataset_name, split='train')
        except:
            return self._load_local(dataset_name)
            
    def _load_local(self, name: str):
        with open(f'data/{name}/train.json') as f:
            return json.load(f)
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def download_datasets():
    api = KaggleApi()
    api.authenticate()
    
    datasets = {
        'MATH': 'competitions/MATH',
        'gsm8k': 'datasets/jeffheaton/gsm8k',
        'theoremqa': 'datasets/wenhuchen/theoremqa'
    }
    
    for name, path in datasets.items():
        if not os.path.exists(f'data/{name}'):
            api.dataset_download_files(path, path=f'data/{name}', unzip=True)
            print(f'Downloaded {name} dataset')
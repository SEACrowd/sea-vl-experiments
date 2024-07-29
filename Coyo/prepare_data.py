from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
import datasets
import pandas as pd

ds = load_dataset("kakaobrain/coyo-700m")['train']

sample_size = len(ds)
country_list = {'kh','th','la','id','mm','ph','vn','tl','bn','my','sg'}
saved_data = []
for idx in tqdm(range(sample_size)):
    temp_data = ds[idx]
    temp_url = temp_data['url']
    if temp_url[:4] == 'http':
        temp_url = temp_url.split('://')[1].split('/')[0].split('.')[-1]
    else:
        temp_url = temp_url.split('/')[0].split('.')[-1]
    if temp_url in country_list:
        temp_data['short_url'] = temp_url
        saved_data.append(temp_data)
        
data_final = datasets.Dataset.from_pandas(pd.DataFrame(data=saved_data))
del saved_data
data_final.save_to_disk('ALL_DATA_filtered')

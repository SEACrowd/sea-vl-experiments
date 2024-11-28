import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_CACHE"]="/workspace/cache"
os.environ["HF_DATASETS_CACHE"]="/workspace/cache"
os.environ["WANDB_DISABLED"] = "true"

import datasets
from datasets import load_dataset, Image
# from distfuse import DistFuse
from tqdm import tqdm
import numpy as np
import pickle
from glob import glob
from PIL import Image
import pandas as pd



file_all = glob("sea-concept-crawler/out/*")
all_title = []
for f in file_all:
    titles = pd.read_csv(f).title.values.tolist()
    all_title+=titles
all_title = list(set(all_title))

coyo_dataset_full = datasets.load_from_disk('/workspace/sea_vl/ALL_DATA_filtered')


all_selected_caption = []

for data in tqdm(coyo_dataset_full):
    for keyword in all_title:
        if keyword.lower() in data['text'].lower():
            all_selected_caption.append([data['text'],data['url']])
            break
            
with open('full_keyword.pkl', 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(all_selected_caption, file) 
import datasets
from datasets import load_dataset, Image
from sentence_transformers import SentenceTransformer

import os
import requests
from tqdm import tqdm
import pickle 

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except:
        pass
    # except Exception as e:
    #     print(f"Failed to download {url}: {e}")

def coyo_download_images(dataset, image_dir='coyo_downloaded_images'):
    os.makedirs(image_dir, exist_ok=True)
    for split in dataset.keys():
        for idx, row in tqdm(enumerate(dataset[split]), total=len(dataset[split])):
            image_url = row[1]
            image_filename = os.path.join(image_dir, f"{split}_{idx}.jpg")
            download_image(image_url, image_filename)
            
coyo_dataset = datasets.load_from_disk("/workspace/sea_vl/ALL_DATA_filtered")


# final_data = dict()
# limit = 100
# for data in tqdm(coyo_dataset):
#     country = data['short_url']
#     if country not in final_data:
#         final_data[country] = [data['text']]
#     else:
#         if len(final_data[country]) < limit and len(data['url']) > 5:
#             final_data[country].append([data['text'],data['url']])
            
# with open('coyo_filtered_1k.pkl', 'wb') as file: 
#     # A new file will be created 
#     pickle.dump(final_data, file) 

coyo_download_images(coyo_dataset)

# sea_vqa = load_dataset('wit543/sea-vqa')

# sea_vqa_download_images(sea_vqa,download=False)
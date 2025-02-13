AUTH_TOKEN=''

from datasets import load_dataset

dataset = load_dataset('worldcuisines/food-kb', '', split='main')

def sea_filter(row):
    SEA_REGION = "South Eastern Asia"
    for i in range(1,6):
        if row[f'region{i}'] == SEA_REGION:
            return True
    return False

dataset = dataset.filter(sea_filter)

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

from pathlib import Path
from tqdm import tqdm

folder = './worldcuisines_images_generated'
Path(folder).mkdir(parents=True, exist_ok=True)

prompt = 'An image of people eating {}'
for k,v in tqdm(enumerate(dataset)):
    image = pipe(
        prompt.format(v['name']),
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator('cpu').manual_seed(0)
    ).images[0]
    image.save(f'{folder}/[{",".join(v["cuisines"])}] {prompt.format(v["name"])}.png')
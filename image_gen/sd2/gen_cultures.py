AUTH_TOKEN=''

import json

with open('../culture_img_gen_prompt_list.json') as f:
    cultures = json.load(f)

import torch
from diffusers import StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

from pathlib import Path
from tqdm import tqdm

folder = '../unesco_cultures_generated/sd2'
Path(folder).mkdir(parents=True, exist_ok=True)

prompt = 'An image of people doing {}'
for data in tqdm(cultures):
    image = pipe(
        prompt.format(data['culture_name']).replace('/', ' or '),
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator('cpu').manual_seed(0)
    ).images[0]
    image.save(f'{folder}/[{data["country"]}] {prompt.format(data["culture_name"]).replace("/", " or ")}.png')
AUTH_TOKEN=''

import json

with open('../landmark_img_gen_prompt_list.json') as f:
    landmarks = json.load(f)

import torch
from diffusers import StableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-large"

pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

from pathlib import Path
from tqdm import tqdm

folder = '../unesco_landmarks_generated/sd3.5'
Path(folder).mkdir(parents=True, exist_ok=True)

prompt = 'An image of people at {}'
for data in tqdm(landmarks):
    image = pipe(
        prompt.format(data['landmark_name']),
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator('cpu').manual_seed(0)
    ).images[0]
    image.save(f'{folder}/[{data["country"]}] {prompt.format(data["landmark_name"])}.png')
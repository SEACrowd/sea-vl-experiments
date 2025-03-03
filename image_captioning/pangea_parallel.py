import sys
import os

llava_path = "/root/LLaVA-NeXT" #git clone https://github.com/LLaVA-VL/LLaVA-NeXT
sys.path.insert(0, llava_path)

os.chdir(llava_path)
print("Working in:", os.getcwd())

import transformers
import torch
import json
from tqdm import tqdm
import re

from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import disable_torch_init
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict
transformers.logging.set_verbosity_error()
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from torch.utils.data import DataLoader 

model_path = 'neulab/Pangea-7B'
model_name = 'Pangea-7B-qwen'
args = {"multimodal": True}
save_path= "/root/vl-script-clean/parallel_result.json"
en_prompt = "Write a caption in English for an image that may include culturally significant objects or elements from Southeast Asia. The caption should specifically name Southeast Asian cultural items, such as cuisine, traditions, landmarks, or other related elements if they appear in the image. The caption should be concise, consisting of 3 to 5 sentences."
dataset = load_dataset('/root/sea-vl_coyo', '', split='train')
batch_size = 16

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=1024, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens
    input_ids = []
    source = sources
    if roles[source[0]["from"]] != roles["human"]: source = source[1:]
    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1: _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None: _input_id = tokenizer(role).input_ids + nl_tokens
            else: _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
    input_ids.append(input_id)
    return torch.tensor(input_ids, dtype=torch.long)


def get_caption(image, prompt):
    image_tensors = []
    prompt = "<image>\n" + prompt

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensors.append(image_tensor.half().cuda())
    input_ids = preprocess_qwen([{'from': 'human', 'value': prompt},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
    
    with torch.inference_mode():
        output_ids = accelerator.unwrap_model(model).generate(
            input_ids,
            images=image_tensors,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs

def save_to_json(save_path, json_obj):
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            json.dump([json_obj], f, indent=4)
    else:
        with open(save_path, "r") as f:
            data = json.load(f)
        data.append(json_obj)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)  

accelerator = Accelerator(
    mixed_precision="fp16",
    device_placement=True
) 


with accelerator.main_process_first(): 
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        None, 
        model_name, 
        torch_dtype="float16",
        # device_map="cuda:0",
        device_map=None,
        **args
    )


dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda x: x 
)
model, dataloader = accelerator.prepare(model,dataloader)  


for batch in tqdm(dataloader, desc="Processing"):
    batch_results = []
    
    for row in batch:
        id_ = row["id"]
        image = row["image"]
        with accelerator.autocast():
            caption = get_caption(image, en_prompt)
            caption = caption.replace('Caption:', '').replace('"','').replace('The caption for this image could be:','')
            batch_results.append({
                "id": id_,
                "caption": caption
            })
    
    gathered_results = accelerator.gather_for_metrics(batch_results)
    
    if accelerator.is_main_process:
        for result in gathered_results:
            save_to_json(save_path, result)
                
    del batch, batch_results, gathered_results
    torch.cuda.empty_cache()

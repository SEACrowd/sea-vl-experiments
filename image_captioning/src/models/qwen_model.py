import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.base_model import BaseModel
from config.settings import Config
from utils.image_utils import load_image

class QwenVL2Model(BaseModel):
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_path = Config.QWENVL_CONFIGURATION["model_path"]
        self.min_pixels = Config.QWENVL_CONFIGURATION["min_pixels"]
        self.max_pixels = Config.QWENVL_CONFIGURATION["max_pixels"]

    def load(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def generate_caption(self, image, prompt, **generation_params):
        image = load_image(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **generation_params
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
            
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    @property
    def name(self):
        return "Qwen2-VL-7B-Instruct"
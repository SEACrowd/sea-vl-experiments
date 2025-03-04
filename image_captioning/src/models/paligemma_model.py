import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from config.settings import Config
from models.base_model import BaseModel
from utils.image_utils import load_image

class PaliGemma2Model(BaseModel):
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_path = Config.PALIGEMMA2_CONFIGURATION["model_path"]

    def load(self):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_path
        )

    def generate_caption(self, image, prompt, **generation_params):
        image = load_image(image)
        model_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                **generation_params
            )
            generation = generation[0][input_len:]
        
        return self.processor.decode(generation, skip_special_tokens=True)

    @property
    def name(self):
        return "paligemma2-10b-ft-docci-448"
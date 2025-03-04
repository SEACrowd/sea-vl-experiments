import torch
from PIL import Image
from models.base_model import BaseModel
from typing import Dict
from utils.image_utils import load_image
from config.settings import Config
import transformers
transformers.logging.set_verbosity_error()

class PangeaModel(BaseModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.model_base = Config.PANGEA_CONFIGURATION["model_base"]
        self.model_path = Config.PANGEA_CONFIGURATION["model_path"]
        self.multimodal = Config.PANGEA_CONFIGURATION["multimodal"]
        self.model_name = Config.PANGEA_CONFIGURATION["model_name"]

    def load(self):
        from llava.model.builder import load_pretrained_model
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=self.model_name,
            torch_dtype="float16",
            multimodal=self.multimodal
        )

    def generate_caption(self, image, prompt, **generation_params):
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from .pangea_utils import preprocess_qwen

        image_tensors = []
        prompt = "<image>\n" + prompt
        image = load_image(image)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensors.append(image_tensor.half().cuda())
        
        input_ids = preprocess_qwen(
            [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': None}],
            self.tokenizer,
            has_image=True
        ).cuda()
        
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensors,
                **generation_params
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip().replace('Caption:', '').replace('"','').replace('The caption for this image could be:','')

    @property
    def name(self):
        return "Pangea-7B"
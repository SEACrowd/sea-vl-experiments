import torch
from PIL import Image
from models.base_model import BaseModel
from config.settings import Config
from utils.image_utils import load_image

class MayaModel(BaseModel):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.model_base = Config.MAYA_CONFIGURATION["model_base"]
        self.model_path = Config.MAYA_CONFIGURATION["model_path"]
        self.mode = Config.MAYA_CONFIGURATION["mode"]

    def load(self):
        from llava.eval.maya.eval_utils import load_maya_model
        self.model, self.tokenizer, self.image_processor, _ = load_maya_model(
            self.model_base,
            self.model_path,
            projector_path if self.mode == "pretrained" else None,
            self.mode
        )
        self.model = self.model.half().cuda().eval()

    def generate_caption(self, image, prompt, **generation_params):
        from llava import conversation, constants
        from llava.mm_utils import process_images, tokenizer_image_token

        if self.model.config.mm_use_im_start_end:
            prompt = f"{constants.DEFAULT_IM_START_TOKEN}{constants.DEFAULT_IMAGE_TOKEN}{constants.DEFAULT_IM_END_TOKEN}\n{prompt}"
        else:
            prompt = f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}"

        conv = conversation.conv_templates["aya"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

    
        input_ids = tokenizer_image_token(
            conv.get_prompt(), 
            self.tokenizer, 
            constants.IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        ).unsqueeze(0).cuda()

        image = load_image(image)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                **generation_params
            )
            
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    @property
    def name(self):
        return "maya"
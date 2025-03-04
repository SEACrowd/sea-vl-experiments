from pathlib import Path
from PIL import Image
from typing import Union

def load_image(image: Union[Path, str, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    elif isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    raise ValueError("Unsupported type for image. Must be a file path (str or Path) or a PIL Image object.")
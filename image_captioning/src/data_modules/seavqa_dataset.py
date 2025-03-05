import json
from pathlib import Path
from typing import Iterator, Dict, Any
from config.settings import Config
from data_modules.base_dataset import BaseDataset

class SEAVQADataset(BaseDataset):
    def __init__(self):
        self.root_dir = Path(Config.SEAVQA_CONFIGURATION["dataset_path"])
        self.images_dir = self.root_dir / "images"

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for json_file in self.root_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            
            country = json_file.stem
            for item in data:
                img_path = self._find_image(item["image_path"])
                if img_path:
                    yield {
                        "country": country,
                        "data": item,
                        "image": img_path
                    }

    def _find_image(self, image_path: str) -> Path:
        target = Path(image_path).name
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_path in self.images_dir.rglob(ext):
                if img_path.name == target:
                    return img_path
        return None

    def filter_items(self, item: Dict[str, Any]) -> bool:
        return True

    @property
    def name(self) -> str:
        return "seavqa"
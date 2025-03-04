from datasets import load_dataset
from typing import Iterator, Dict, Any
from config.settings import Config
from data_modules.base_dataset import BaseDataset

class WorldCuisinesDataset(BaseDataset):
    def __init__(self):
        self.dataset = load_dataset(Config.WORLDCUISINE_CONFIGURATION["dataset_path"], '', split='main')
        self.dataset = self.dataset.filter(self.filter_items)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            for i in range(1, Config.WORLDCUISINE_CONFIGURATION["MAX_IMAGES_PER_ITEM"] + 1):
                img_key = f"image{i}"
                if item.get(img_key):
                    yield {
                        "data": item,
                        "image": item[img_key],
                        "image_url": item.get(f"{img_key}_url", ""),
                    }

    def filter_items(self, item) -> bool:
        return any(item.get(f"region{i}") in Config.WORLDCUISINE_CONFIGURATION["SEA_REGIONS"]
               for i in range(1, 6))

    @property
    def name(self) -> str:
        return "worldcuisines/food-kb"
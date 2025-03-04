import json
from pathlib import Path
from typing import List, Dict
from config.settings import Config

class ResultSaver:
    def __init__(self, output_dir: Path = Config.RESULTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_batch(self, results: List[Dict], filename: str):
        output_path = self.output_dir / f"{filename}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    def append_results(self, results: List[Dict], filename: str):
        output_path = self.output_dir / f"{filename}.json"
        existing = []
        if output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)
        existing.extend(results)
        self.save_batch(existing, filename)
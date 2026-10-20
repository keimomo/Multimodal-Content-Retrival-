import argparse
import torch
from pathlib import Path
import yaml
import datetime

class FileConfig:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).expanduser().resolve()
        print(self.project_root)
        self.data_root = self.project_root / "data"
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.queue_dir = self.data_root / "queue"
        self.processed_csv_dir = self.processed_dir / "csv"

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.processed_csv_dir.mkdir(parents=True, exist_ok=True)

        self.save_config()
    
    def save_config(self):
        config = {
           "project":{
                "name": "lecture_rag",
                "description": "A project for lecture RAG",
                "author": "Kei Matsui",
                "version": "0.1.0",
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },

            "paths": {
                "root": str(self.project_root),
                "raw": str(self.raw_dir),
                "processed": str(self.processed_dir),
                "queue": str(self.queue_dir),
                "processed_csv": str(self.processed_csv_dir),
            },

            "device": self.device,
        }
        with open(Path(self.project_root / "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        
        print(f"Config saved to {Path(self.project_root / 'config.yaml')}")

    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--root", type=str, required=False, help="A project root to donwload a video. If not provided, the current working directory will be used.")
    args = parse.parse_args()

    if args.root is not None:
        root = Path(args.root)
    else:
        root = Path.cwd()

    FileConfig(str(root))

    
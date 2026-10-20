import yaml
from audio_processor import AudioProcessor
from pathlib import Path
import argparse

class RawProcess:
    def __init__(self, project_root: Path):
        with open(str(project_root / "config.yaml"), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.raw_dir = Path(config["paths"]["raw"])
        self.audio_processor = AudioProcessor("medium", config["device"])


    def process_raw(self):
        for file in self.raw_dir.glob("*"):
            if file.is_file():
                if file.suffix == ".mp4" or file.suffix == ".mp3":
                    audio_data = self.audio_processor.transcribe(file)
                    print(audio_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    path = Path(args.root).expanduser().resolve()
    print(f"Processing raw data from {path}")
    raw_process = RawProcess(path)
    raw_process.process_raw()
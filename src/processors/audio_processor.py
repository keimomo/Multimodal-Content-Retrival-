import whisper
from pathlib import Path
import gc
import torch
import json
class AudioProcessor:
    def __init__(self, queue_dir:Path, model_name: str = "small", device: str = "cpu"):
        self.device = "cuda" if device == "cuda" else "cpu"
        self.model = None
        self.model_name = model_name
        self.queue_dir = queue_dir
    
    def save_json(self, path: Path, value: dict):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def transcribe(self, file_path: Path,  modality: str):
        video_id = file_path.stem
        result = self.model.transcribe(str(file_path), word_timestamps=True, fp16=(self.device == "cuda"), language="ja")
        metadata = {
            "modality": modality,
            "video_id": video_id,
            "text": result
        }
        output_path = Path(self.queue_dir / f"{video_id}.json")
        self.save_json(output_path, metadata)
        return output_path
    
    def load_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_name, device=self.device)

    def unload_model(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
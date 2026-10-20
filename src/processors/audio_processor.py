import whisper
from pathlib import Path
import gc
import torch

class AudioProcessor:
    def __init__(self, model_name: str = "small", device: str = "cpu"):
        self.device = "cuda" if device == "cuda" else "cpu"
        self.model = None
        self.model_name = model_name
    

    def transcribe(self, file_path: Path, modality: str):
        video_id = file_path.stem
        result = self.model.transcribe(str(file_path), word_timestamps=True, fp16=(self.device == "cuda"), language="ja")
        metadata = {
            "file_path": file_path,
            "modality": modality,
            "video_id": video_id,
            "text": result
        }
        return metadata
    
    def load_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_name, device=self.device)

    def unload_model(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
import whisper
import yaml
from pathlib import Path
class AudioProcessor:
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        self.device = "cuda" if device == "cuda" else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
    
    def transcribe(self, file_path: Path):

        result = self.model.transcribe(str(file_path), word_timestamps=True, fp16=(self.device == "cuda"), language="ja")
        metadata = {
            "file_path": file_path,
            "modality": "audio_text",
            "text": {
                "full_text": result["text"],
                "segments": [f"{segment['start']:.2f}_{segment['end']:.2f}_{segment['text']}" for segment in result["segments"]]
            }
        }
        return metadata
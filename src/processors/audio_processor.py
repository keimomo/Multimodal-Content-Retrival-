import whisper
from pathlib import Path
import gc
import torch
class AudioProcessor:
    def __init__(self, device: torch.device):
        self.model = None
        self.model_name = "small"
        self.device = torch.device("cpu") if device != torch.device("cuda") else device # whisper only supports cpu and cuda. mpu -> cpuu
    
    

    def transcribe(self, file_path: str):
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        if Path(file_path).suffix != ".mp4":
            raise ValueError("Only video files are supported. Please provide a .mp4 file.")
        return self.model.transcribe(file_path, word_timestamps=False, fp16=(self.device == torch.device("cuda")))
    
    def load_model(self, name: str = "small"):
        if self.model is None:
            self.model = whisper.load_model(name, device=self.device)
            self.model_name = name
            print(f"Model {self.model_name} loaded successfully.")

    def unload_model(self):
        print(f"Unloading model {self.model_name}...")
        self.model = None
        gc.collect()
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
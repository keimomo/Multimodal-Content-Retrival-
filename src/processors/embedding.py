from sentence_transformers import SentenceTransformer
import torch
import gc
import numpy as np
from pathlib import Path

class MultiModalEmbedding:
    def __init__(self,device: torch.device, model_name: str = "google/embeddinggemma-2"):
        self.model_name = model_name
        self.device = device

        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"[load] Embeddings: {self.model_name} on {self.device}; weights may download on first use...")
            if self.device == torch.device("cpu"):
                dtype = torch.float32
            else:
                dtype = torch.float16
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"torch_dtype": dtype},
                config_kwargs={"audio_config": None},
            )
            print(f"[ready] Embedding model loaded: {self.model_name}")
    def unload_model(self):
        print(f"[memory] Releasing embedding model: {self.model_name}")
        self.model = None
        gc.collect()
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()

    def embed_text(self, text: str) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False,
        ).reshape(-1)
        self.release_cache()
        return embedding

    def embed_video(self, file_path: Path) -> np.ndarray:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Video clip not found: {file_path}")
        if self.model is None:
            raise ValueError("Model not loaded.")

        embedding = self.model.encode(
            {"video": str(file_path)},
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False,
            processing_kwargs={
                "video": {
                    "fps": 1,
                    "max_frames": 32,
                    "add_timestamps": True,
                    "max_soft_tokens": 280,
                    "overflow_strategy": "uniform",
                }
            },
        ).reshape(-1)
        self.release_cache()

        return embedding

    def release_cache(self):
        gc.collect()
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()

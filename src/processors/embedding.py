from sentence_transformers import SentenceTransformer
import torch
import gc
import numpy as np
from pathlib import Path

class MultiModalEmbedding:
    def __init__(self, model_name: str = "google/embeddinggemma-2"):
        self.model_name = model_name
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.model = None

    def load_model(self):
        if self.model is None:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"torch_dtype": dtype},
                config_kwargs={"audio_config": None},
            )
    def unload_model(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #TODO: Implement text embedding
    def embed_text(self, text: str) -> np.ndarray:
        pass

    def embed_video(self, meta_data: list[dict]) -> list[dict]:
        embedded_meta_data = []
        for data in meta_data:
            file_path = Path(data["video_path"])
            print(f"Embedding video: {file_path.name}")
            embedding = self.model.encode(
                str(file_path),
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device,
                show_progress_bar=True,
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

            embedded_meta_data.append({
                "meta_data": data,
                "embedding": embedding
            })
            file_path.unlink()

        return embedded_meta_data

    def embed_image(self, image: str) -> np.ndarray:
        pass
    
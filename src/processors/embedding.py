from sentence_transformers import SentenceTransformer
import torch
import gc
import numpy as np
from pathlib import Path

class MultiModalEmbedding:
    def __init__(self, model_name: str = "google/embeddinggemma-2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device

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
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def embed_text(self, meta_data: list[dict]) -> list[dict]:
        embedded_meta_data = []
        for chunk in meta_data:
            meta_data = {
                "id": chunk["id"],
                "original_id": chunk["original_id"],
                "modality": chunk["modality"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"]
            }
            embedding = self.model.encode(
                chunk["text"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device,
                show_progress_bar=True,
            ).reshape(-1)
            embedded_meta_data.append({
                "meta_data": meta_data,
                "embedding": embedding
            })
        return embedded_meta_data

    def embed_video(self, file_path: Path) -> np.ndarray:
        embedding = self.model.encode(
            str(file_path),
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

        return embedding

    def embed_image(self, image: str) -> np.ndarray:
        pass
    
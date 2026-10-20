from sentence_transformers import SentenceTransformer
import torch
import numpy as np

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

        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"audio_config": None},
        )

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def embed_video(self, video: str) -> np.ndarray:
        return self.model.encode(video)

    def embed_image(self, image: str) -> np.ndarray:
        return self.model.encode(image)
    
from sentence_transformers import SentenceTransformer

class MultiModalEmbedding:
    def __init__(self, model_name: str = "google/embeddinggemma-2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    
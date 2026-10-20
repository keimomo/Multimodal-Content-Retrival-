import pinecone
from dotenv import load_dotenv
import os
import numpy as np
from pinecone import Pinecone, SchemaBuilder


class PineconeDatabase:
    def __init__(self):
        load_dotenv()
        self.db = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        schema = (
            SchemaBuilder()
            .add_dense_vector_field(name="embedding", dimension=768, metric="cosine")
            .build()
        )

        if not self.db.indexes.exists("video-vectors-dev"):
            print("Creating index...")
            self.db.indexes.create(
                name="video-vectors-dev",
                schema=schema
            )
            print("video-vectors-dev index created")

        if not self.db.indexes.exists("text-vectors-dev"):
            print("Creating index...")
            self.db.indexes.create(
                name="text-vectors-dev",
                schema=schema
            )
            print("text-vectors-dev index created")
    
        self.video_index = self.db.Index("video-vectors-dev")
        self.text_index = self.db.Index("text-vectors-dev")

    
    def add_video_data(self, embedding_meta_data: list[dict]):
        try:
            for data in embedding_meta_data:
                embedding = data["embedding"]
                meta_data = data["meta_data"]
                id = f"{meta_data['video_id']}_{meta_data['clip_id']}"
            
                self.video_index.upsert(
                    namespace = "video",
                    vectors = [
                        {
                            "id": id,
                            "values": embedding.tolist(),
                            "metadata": {
                                "modality": "video",
                                "start": meta_data["start"],
                                "end": meta_data["end"],
                            }
                        }
                    ]
                )
                print(f"Video data added to database: {id}")
        except Exception as e:
            print(f"Error adding video data: {e}")

    def add_text_data(self, meta_data: dict, embedding: np.ndarray):
        pass

    def query(self, query):
        pass

    

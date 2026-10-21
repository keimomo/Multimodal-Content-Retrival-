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
            
                self.video_index.documents.upsert(
                    namespace = "video",
                    documents = [
                        {
                            "_id": id,
                            "embedding": embedding.tolist(),
                            "modality": "video",
                            "start": meta_data["start"],
                            "end": meta_data["end"],
                            
                        }
                    ]
                )
                print(f"Video data added to database: {id}")
        except Exception as e:
            print(f"Error adding video data: {e}")

    def add_text_data(self, meta_data: list[dict]):
        try:
            for data in meta_data:
                embedding = data["embedding"]
                meta_data = data["meta_data"]
                id = f"{meta_data['original_id']}_{meta_data['id']}"
                
                self.text_index.documents.upsert(
                    namespace = "text",
                    documents = [
                        {
                            "_id": id,
                            "original_id": meta_data["original_id"],
                            "id": meta_data["id"],
                            "embedding": embedding.tolist(),
                            "modality": meta_data["modality"],
                            "start": meta_data["start"],
                            "end": meta_data["end"],
                            "text": meta_data["text"],
                        }
                    ]
                )
                print(f"Text data added to database: {id}")
        except Exception as e:
            print(f"Error adding text data: {e}")

    def query(self, query):
        pass

    

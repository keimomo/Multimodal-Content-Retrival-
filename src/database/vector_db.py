from pinecone import Pinecone, SchemaBuilder, DenseVectorQuery
import numpy as np

class PineconeDatabase:
    def __init__(self, get_api_key: callable):
        self.db = Pinecone(api_key=get_api_key("pinecone"))
        self.index = None

    def connect(self):
        schema = (
            SchemaBuilder()
            .add_dense_vector_field(name="embedding", dimension=768, metric="cosine")
            .build()
        )

        if not self.db.indexes.exists("mm-vectors-cli"):
            print("Creating index...")
            self.db.indexes.create(
                name="mm-vectors-cli",
                schema=schema
            )
            print("mm-vectors-cli index created")

        self.index = self.db.Index("mm-vectors-cli")

    def add_video_data(self, name, embedding, start, end, source_id, source_path):
        try:
            self.index.documents.upsert(
                namespace = "video",
                documents = [
                    {
                        "_id": name,
                        "source": source_id,
                        "source_path": source_path,
                        "embedding": embedding.tolist(),
                        "start": start,
                        "end": end,
                    }
                ]
            )
            print(f"Video data added to database: {id}")
        except Exception as e:
            print(f"Error adding video data: {e}")

    def add_text_data(self, embedding, data_id: int, source_path: str, chunk_id: int, start: float, end: float):
        try:  
            self.index.documents.upsert(
                namespace = "text",
                documents = [
                    {
                        "_id": str(chunk_id),
                        "source": data_id,
                        "source_path": source_path,
                        "embedding": embedding.tolist(),
                        "modality": "transcription_chunks",
                        "start": start,
                        "end": end,
                    }
                ]
            )
        except Exception as e:
            print(f"Error adding text data: {e}")

    def query(self, query):
        visual_serach = self.index.documents.search(
            namespace="videos",
            top_k=10,
            score_by=[DenseVectorQuery(
                field="embedding",
                values=np.array(query).reshape(-1).tolist()
            )],
            include_fields=["*"]
        )
        text_serach = self.index.documents.search(
                namespace="text",
                top_k=10,
                score_by=[DenseVectorQuery(
                    field="embedding",
                    values=np.array(query).reshape(-1).tolist()
                )],
                include_fields=["*"]
            )

        video_hits = [{
            "score": hit.score,
            "source": hit.source,
            "source_path": hit.source_path,
            "start": hit.start,
            "end": hit.end,
            "modality": "video",
        } for hit in visual_serach.matches]

        text_hits = [{
            "score": hit.score,
            "chunk_id": hit.id,
            "source": hit.source,
            "source_path": hit.source_path,
            "start": hit.start,
            "end": hit.end,
            "modality": "text",
        } for hit in text_serach.matches]

        hits = video_hits+text_hits
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        return hits[:5]
        

    

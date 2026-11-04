from pinecone import Pinecone, SchemaBuilder


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

    def disconnect(self):
        self.index = None

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

    

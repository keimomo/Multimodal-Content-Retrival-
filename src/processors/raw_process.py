from processors.embedding import MultiModalEmbedding

# from ..database.sql_db import SQLDB
# from ..database.vector_db import PineconeDatabase

import subprocess
import os
from pathlib import Path
import shutil
import pandas as pd

class RawProcess:
    def __init__(self, device: str = "cuda"):
        #embedding initialization
        self.embedding = MultiModalEmbedding(device=device)
    
    # # process all raw data from raw directory and move it to processed directory
    # def process_text(self):
        
    def process_video(self, connection, sql_db, pinecone_db, abs_path: str, data_id: str,):
        #1 create 30s clips of video
        data_path = self.chunking_video(abs_path)

        #get clip metadata and save it in the sql database
        df = pd.read_csv(data_path / f"segments.csv", header=None, names=["name", "start", "end"])
        # for each clip, generate embedding and save it in the array. 

        for _, row in df.iterrows():
            name = row["name"]
            start = row["start"]
            end = row["end"]
            sql_db.insert_chunk(connection=connection, source_id=data_id, modality="video", start_time=start, end_time=end, status="completed", text=None)
            embedding = self.embedding.embed_video(file_path=Path(data_path / name))
            pinecone_db.add_video_data(name=name, embedding=embedding, start=start, end=end, source_id=data_id, source_path=abs_path)

        #delete temp file
        shutil.rmtree(data_path)
        sql_db.update_source_status(connection=connection, id=data_id, status="completed")




    def chunking_video(self, abs_path: str, clip_duration:int = 30):
        temp_file = Path(abs_path).parent / f"temp"
        os.makedirs(temp_file, exist_ok=True)
        
        pattern = temp_file / f"%06d.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-i", abs_path, #input file
            "-f", "segment", #format
            "-segment_time", str(clip_duration), #segment time
            "-reset_timestamps", "1", #reset timestamps
            "-c", "copy", #copy
            "-segment_list", str(temp_file / f"segments.csv"),
            "-segment_list_type", "csv",
            str(pattern), #output pattern
        ]
        subprocess.run(command, check=True)
        
        return temp_file
    # def process_queue(self):
    #     try:
    #         print("Processing queue...")
    #         self.embedding.load_model()
    #         while self.visual_meta_data:
    #             meta_data = self.visual_meta_data.popleft()
    #             embedding_meta_data = self.embedding.embed_video(meta_data)
    #             self.db.add_video_data(embedding_meta_data)
    #         tokenizer = self.embedding.model.tokenizer
    #         def count_tokens(text):
    #             return len(tokenizer.encode(text, add_special_tokens=False, truncation=True))
            
    #         budget = 500
    #         overlap = 100
    #         while self.audio_meta_data:
    #             json_path = self.audio_meta_data.popleft()
    #             chunks = self.text_processor.process_audio_transcription(json_path, count_tokens, budget, overlap)
    #             json_path.rename(self.processed_dir / json_path.name)
    #             embedded_meta_data = self.embedding.embed_text(chunks)
    #             self.db.add_text_data(embedded_meta_data)
    #         # while not self.text_meta_data.empty():
    #         #     meta_data = self.text_meta_data.popleft()
    #         #     embedding_meta_data = self.embedding.embed_text(meta_data)
    #         #     self.db.add_text_data(embedding_meta_data)
    #         print("Queue processing completed")
    #     except Exception as e:
    #         print(f"Error processing queue: {e}")
    #     finally:
    #         self.embedding.unload_model()

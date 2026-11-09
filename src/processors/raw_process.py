from processors.embedding import MultiModalEmbedding
import torch

import subprocess
import os
from pathlib import Path
import shutil
import pandas as pd
import re
import json

class RawProcess:
    def __init__(self, device: torch.device):
        #embedding initialization
        self.embedding = MultiModalEmbedding(device=device)
    
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

    def count_tokens(self, text: str, budget: int = 500, overlap: int = 100):
        tokenizer = self.embedding.model.tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=True))

    def process_text(self, connection, sql_db, pinecone_db, abs_path: str, data_id: str,):
        #1 get transcription text/json from db
        row = sql_db.get_transcription(connection, data_id)

        if row is None:
            raise ValueError(f"Transcription not found for source with id {data_id}.")

        transcript = json.loads(row[0])["segments"]
        chunks = self.build_chunks(transcript, data_id, self.count_tokens)
        #2 chunk them and store chunk metadata indb<- prob better to use queue or stack or some sort so mem is optimized
        while chunks:
            chunk = chunks.pop()
            embedding = self.embedding.embed_text(chunk["text"])
            chunk_id = sql_db.insert_chunk(connection=connection, source_id=data_id, modality="transcription_chunks", start_time=chunk["start"], end_time=chunk["end"], status="completed", text=chunk["text"])
            pinecone_db.add_text_data(embedding=embedding, data_id=data_id, source_path=abs_path, chunk_id=chunk_id, start=chunk["start"], end=chunk["end"])
        sql_db.update_source_status(connection=connection, id=data_id, status="text_embeded")
    
    # clean text to remove extra spaces and newlines
    def clean_text(self, text: str):
            return re.sub(r"\s+", " ", text).strip()

    # split text into chunks based on token budget and overlap
    #recurcively split the text into chunks -> return a list of chunks
    def split_text(self, text:str, count_tokens, max_tokens: int):
        # no need to split the given text if its under the budget
        if count_tokens(text) <= max_tokens:
            return [text]

        # find the middle of the text and split it at the first space after the middle
        middle = len(text) // 2
        spaces = [m.start() for m in re.finditer(r"\s+", text)]
        cut = min(spaces, key=lambda x: abs(x - middle)) if spaces else middle

        if not 0 < cut < len(text):
            raise ValueError("Token budget is too small for one character.")
            
        return (self.split_text(text[:cut].strip(), count_tokens, max_tokens)
                        + self.split_text(text[cut:].strip(), count_tokens, max_tokens))
        
    def build_chunks(self, segments, id, count_tokens, budget:int = 500, overlap: int = 100):
        if not 0 <= overlap < budget:
            raise ValueError("Overlap must be between 0 and budget.")
            
        units = []
        for i, segment in enumerate(segments):
            text = self.clean_text(segment["text"])
            if not text:
                continue

            start = float(segment["start"])
            end = float(segment["end"])
            
            for piece in self.split_text(text, count_tokens, budget):
                units.append({
                    "text": piece,
                    "start": start,
                    "end": end,
                    "segment_id": segment.get("id", i),
                })

        chunks = []
        i = 0
        chunk_id = 0
        while i < len(units):
            j = i
            text = ""
            while j< len(units):
                combined_text = (text+ " " + units[j]["text"]).strip()
                if count_tokens(combined_text) <= budget:
                    text = combined_text
                    j += 1
                else:
                    break

            if i == j:
                raise ValueError("Token budget is too small for one character.")
                
            selected_chunk = units[i:j]
            chunks.append({
                "id": f"{id}_{chunk_id}",
                "modality": f"transcription_chunks",
                "start": selected_chunk[0]["start"],
                "end": selected_chunk[-1]["end"],
                "text": text,
            })
            chunk_id += 1
            if j == len(units):
                break
                
            temp = j
            while temp > i + 1:
                suffix = " ".join(u["text"] for u in units[temp - 1:j])
                if count_tokens(suffix) > overlap:
                    break
                temp -= 1
            i = temp

        return chunks

    def semantic_search(self, pinecone_db, user_query:str):
        embedding = self.embedding.embed_text(user_query)
        hits = pinecone_db.query(embedding)
        hits = [h for h in hits if h["score"] > 0.5]
        return hits
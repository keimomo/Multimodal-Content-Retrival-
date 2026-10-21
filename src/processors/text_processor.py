from pathlib import Path
import re
import json




class TextProcessor:
    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
    
    def clean_text(self, text: str):
        return re.sub(r"\s+", " ", text).strip()

    def split_text(self, text:str, count_tokens, max_tokens: int):
        if count_tokens(text) <= max_tokens:
            return [text]
        
        middle = len(text) // 2
        spaces = [m.start() for m in re.finditer(r"\s+", text)]
        cut = min(spaces, key=lambda x: abs(x - middle)) if spaces else middle

        if not 0 < cut < len(text):
            raise ValueError("Token budget is too small for one character.")
        
        return (self.split_text(text[:cut].strip(), count_tokens, max_tokens)
                    + self.split_text(text[cut:].strip(), count_tokens, max_tokens))
     
    def build_chunks(self, segments, id, modality,count_tokens, budget:int = 220, overlap: int = 40):
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
                "id": f"{id}_transcript_{chunk_id}",
                "original_id": id,
                "modality": f"{modality}_transcript",
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

    def load_json(self, file: Path):
        with open(file, "r") as f:
            return json.load(f)

    def process_audio_transcription(self, file: Path, count_tokens, budget:int = 500, overlap: int = 100):
        metadata = self.load_json(file)
        chunks = self.build_chunks(metadata["text"]["segments"], metadata["video_id"], metadata["modality"], count_tokens, budget, overlap)
        return chunks
        

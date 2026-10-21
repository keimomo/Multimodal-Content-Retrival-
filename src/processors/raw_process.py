from ast import Pass
import yaml

from processors.audio_processor import AudioProcessor
from processors.video_processor import VideoProcessor
from processors.text_processor import TextProcessor
from processors.embedding import MultiModalEmbedding

from pathlib import Path
import argparse
from database.database import PineconeDatabase
from collections import deque

class RawProcess:
    def __init__(self, project_root: Path, db: PineconeDatabase):
        with open(str(project_root / "config.yaml"), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        #usefull paths
        self.raw_dir = Path(config["paths"]["raw"])
        self.processed_dir = Path(config["paths"]["processed"])
        self.queue_dir = Path(config["paths"]["queue"])
        self.csv_dir = Path(config["paths"]["processed_csv"])

        #processor initialization
        self.audio_processor = AudioProcessor(self.queue_dir, "small", config["device"])
        self.video_processor = VideoProcessor(self.csv_dir, self.queue_dir)
        self.text_processor = TextProcessor(self.queue_dir)

        #embedding initialization
        self.embedding = MultiModalEmbedding()

        # database refference
        self.db = db

        #data queues - global data storage for processing: later replace it with databse when microservices are implemented
        self.visual_meta_data = deque()
        self.text_meta_data = deque()
        self.audio_meta_data = deque()

    def run(self):
        self.process_raw()
        self.process_queue()

        self.visual_meta_data.clear()
        self.audio_meta_data.clear()
        self.text_meta_data.clear()
        print("Raw processing completed")
    
    # process all raw data from raw directory and move it to processed directory
    def process_raw(self):
        try:
            self.audio_processor.load_model()
            for file in self.raw_dir.glob("*"):
                if file.suffix == ".mp4":
                    self.visual_meta_data.append(self.video_processor.video_processing(file))
                    self.audio_meta_data.append(self.audio_processor.transcribe(file, "video"))
                elif file.suffix == ".mp3":
                    self.audio_meta_data.append(self.audio_processor.transcribe(file, "audio"))
                elif file.suffix == ".txt":
                    pass #TODO: Implement text processing
                    # self.text_meta_data.append(self.text_processor.process_text(file))
                file.rename(self.processed_dir / file.name)
            print("Raw processing completed")
        except Exception as e:
            print(f"Error processing raw data: {e}")
        finally:
            self.audio_processor.unload_model()
        

    def process_queue(self):
        try:
            print("Processing queue...")
            self.embedding.load_model()
            while self.visual_meta_data:
                meta_data = self.visual_meta_data.popleft()
                embedding_meta_data = self.embedding.embed_video(meta_data)
                self.db.add_video_data(embedding_meta_data)
            tokenizer = self.embedding.model.tokenizer
            def count_tokens(text):
                return len(tokenizer.encode(text, add_special_tokens=False, truncation=True))
            
            budget = 500
            overlap = 100
            while self.audio_meta_data:
                json_path = self.audio_meta_data.popleft()
                chunks = self.text_processor.process_audio_transcription(json_path, count_tokens, budget, overlap)
                json_path.rename(self.processed_dir / json_path.name)
                embedded_meta_data = self.embedding.embed_text(chunks)
                self.db.add_text_data(embedded_meta_data)
            # while not self.text_meta_data.empty():
            #     meta_data = self.text_meta_data.popleft()
            #     embedding_meta_data = self.embedding.embed_text(meta_data)
            #     self.db.add_text_data(embedding_meta_data)
            print("Queue processing completed")
        except Exception as e:
            print(f"Error processing queue: {e}")
        finally:
            self.embedding.unload_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    path = Path(args.root).expanduser().resolve()
    print(f"Processing raw data from {path}")
    raw_process = RawProcess(path)
    raw_process.process_raw()
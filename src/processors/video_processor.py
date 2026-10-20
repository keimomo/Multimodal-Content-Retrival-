import uuid 
from pathlib import Path
import pandas as pd
import subprocess

class VideoProcessor:
    def __init__(self, csv_dir: Path, queue_dir: Path, clip_duration: int = 30):
        self.clip_duration = clip_duration
        self.csv_dir = csv_dir
        self.queue_dir = queue_dir

    def csv_processing(self, data_id: str):
        df = pd.read_csv(Path(self.csv_dir / f"{data_id}_segments.csv"), header=None, names=["name", "start", "end"])
        video_meta_data = []
        for index, row in df.iterrows():
            video_id, _ = row["name"].split("_")
            video_path = f"{self.queue_dir}/{row["name"]}"
            video_data = {
                "video_id": video_id,
                "clip_id": index,
                "modality": "video",
                "video_path": video_path,
                "start": row["start"],
                "end": row["end"]
            }
            video_meta_data.append(video_data)
        return video_meta_data
    
    def chunking_video(self, file: Path):
        
        data_id = file.stem
        pattern = self.queue_dir / f"{data_id}_%06d{file.suffix}"
        command = [
            "ffmpeg",
            "-y",
            "-i", str(file), #input file
            "-f", "segment", #format
            "-segment_time", str(self.clip_duration), #segment time
            "-reset_timestamps", "1", #reset timestamps
            "-c", "copy", #copy
            "-segment_list", str(self.csv_dir / f"{data_id}_segments.csv"),
            "-segment_list_type", "csv",
            str(pattern), #output pattern
        ]
        subprocess.run(command, check=True)
        return data_id

    def video_processing(self, file: Path):
        data_id = self.chunking_video(file)
        meta_data = self.csv_processing(data_id)
      
        return meta_data


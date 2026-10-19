import uuid 
from pathlib import Path
import pandas as pd

class VideoProcessor:
    def __init__(self, input_dir: Path, output_dir: Path, clip_duration: int = 30):
        self.clip_duration = clip_duration
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.csv_dir = output_dir / "csv"
        self.processed_file = []

   def csv_processing(self):
        video_data = []
        for csv_file in self.csv_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, header=None, names=["name", "start", "end"])
            for index, row in df.iterrows():
                video_id, _ = row["name"].split("_")
                video_path = f"{self.csv_dir}/{row["name"]}"
                video_data = {
                    "video_id": video_id,
                    "clip_id": index,
                    "modality": "video",
                    "video_path": video_path,
                    "start": row["start"],
                    "end": row["end"]
                }
                video_data.append(video_data)
            self.processed_file.append(csv_file)
        return video_data
    
    def chunking_video(self):
        source = Path(self.input_dir)
        output = Path(self.output_dir)
        for file in source.glob("*.mp4"):
            data_id = uuid.uuid4()
            pattern = output / f"{data_id}_%06d{file.suffix}"
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
      
    def clear_files(self):
        for file in self.processed_file:
            file.unlink()
        self.processed_file = []

    def video_processing(self, clear_files: bool = True):
        self.chunking_video()
        meta_data = self.csv_processing()
        if clear_files:
            self.clear_files()
        return meta_data


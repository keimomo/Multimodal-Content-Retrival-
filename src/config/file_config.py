import argparse
import subprocess
import uuid
from pathlib import Path
import os

class FileConfig:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).expanduser().resolve()
        print(self.project_root)
        self.data_root = self.project_root / "data"
        self.input_dir = self.data_root / "input"
        self.queue_dir = self.data_root / "queue"
        self.history_dir = self.data_root / "history"
        self.csv_dir = self.history_dir / "csv"

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    """
    Getters
    """
    def get_input_dir(self):
        return self.input_dir

    def get_queue_dir(self):
        return self.queue_dir
    
    def get_history_dir(self):
        return self.history_dir
    
    def get_csv_dir(self):
        return self.csv_dir
    
    def get_project_root(self):
        return self.project_root
    

    """
    Installers
    """
    def install_via_url(self, url: str, video_id: str = None):

        if video_id is None:
            video_id = str(uuid.uuid4())

        try:
            subprocess.run([
                "yt-dlp",
                "--no-playlist",
                "-t", "mp4",
                "-o", str(self.input_dir / f"{video_id}.mp4"),
                url
            ], check=True)
            print(f"Video {video_id} downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading video {video_id}: {e}")

    def install_via_zip(self, zip_file: str):
        zip_path = Path(self.data_root / "zip_files").expanduser().resolve()
        print(zip_path)
        zip_path.mkdir(parents=True, exist_ok=True)

        zip_file = Path(zip_file).expanduser().resolve()
        print(zip_file)
        try:
            subprocess.run([
                "unzip",
                zip_file,
                "-d", str(zip_path)
            ], check=True)
            print(f"Zip file {zip_file} unzipped successfully")
            self.instal_via_directory(zip_path)
        except subprocess.CalledProcessError as e:
            print(f"Error unzipping zip file {zip_file}: {e}")
            subprocess.run([
                "rm",
                "-rf",
                str(zip_path)
            ], check=True)

        subprocess.run([
                "rm",
                "-rf",
                str(zip_path)
            ], check=True)

    def instal_via_directory(self, directory: str):
        directory = Path(directory).expanduser().resolve()
        print(directory)
        media_files = [
            path
            for path in directory.rglob("*.mp4")
            if not path.name.startswith("._")
            and "__MACOSX" not in path.parts
        ]
        
        if len(media_files) == 0:
            print(f"No media files found in directory {directory}")
            return
        
        for media_file in media_files:
            try:
                subprocess.run([
                    "cp",
                    str(media_file),
                    str(self.input_dir)
                ], check=True)
                print(f"Media file {media_file} copied successfully")
            except subprocess.CalledProcessError as e:
                print(f"Error copying media file {media_file}: {e}")
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--root", type=str, required=False, help="A project root to donwload a video. If not provided, the current working directory will be used.")
    args = parse.parse_args()

    if args.root is not None:
        root = Path(args.root)
    else:
        root = Path.cwd()

    file_config = FileConfig(str(root))

    print(f"Ready to install video")
    print(f"To install a video, enter the video url. Example: url https://www.youtube.com/watch?v=1234567890\n")
    print(f"To install a zip file, enter the zip file path. Example: zip /path/to/zip/file.zip\n")
    print(f"To copy movies in a directory, enter the directory path. Example: directory /path/to/directory\n")
    print(f"Enter 'exit' to stop\n")
    while True:
        command = input()
        if command == "exit":
            print("Exiting...")
            break
        elif command.startswith("url"):
            url = command.split(" ")[1]
            file_config.install_via_url(url)
        elif command.startswith("zip"):
            zip_file = command.split(" ")[1]
            file_config.install_via_zip(zip_file)
        elif command.startswith("directory"):
            directory = command.split(" ")[1]
            file_config.instal_via_directory(directory)
        else:
            print("Invalid command")
from pathlib import Path

class TextProcessor:
    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir

    def process_audio_transcription(self, file: Path):
        with open(file, "r") as f:
            audio_transcription = f.read()
        return audio_transcription

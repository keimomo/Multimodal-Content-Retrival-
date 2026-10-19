# Video Moment Retrieval

An experimental project for finding time ranges in videos from natural language. The current work is a Jupyter notebook that explores video chunking, speech transcription with Whisper, and visual embeddings with Google's EmbeddingGemma 2. A broader project proposal is in [Video Moment Retrieval — Project Formation.md](<Video Moment Retrieval — Project Formation.md>).

## Current state

The notebook is the main working entry point: [`notebooks/MM_embedding.ipynb`](notebooks/MM_embedding.ipynb). It contains experiments for splitting videos into roughly 30-second clips, transcribing speech, and encoding video clips. The embedding results are currently assembled in notebook memory; there is no implemented vector index or text-query search interface yet.

The Python package under `src/processors/` is unfinished and should not be treated as a working command-line pipeline. `video_processor.py` currently has syntax and implementation issues, and `embedding.py` only initializes a model. The Dockerfile also refers to a `main.py` that is not present. The project formation document describes planned retrieval and video-LLM approaches; those are goals, not features implemented in this repository.

## Repository contents

- `notebooks/MM_embedding.ipynb` — experimental media processing, Whisper transcription, and EmbeddingGemma 2 embedding workflow.
- `notebooks/note_assets/input/` — sample source video (`vsample1.mp4`).
- `notebooks/note_assets/output/` — generated video clips and CSV segment metadata. These are notebook artifacts, not a search index.
- `src/processors/video_processor.py` — in-progress video chunking and metadata processor.
- `src/processors/embedding.py` — in-progress wrapper for `SentenceTransformer`.
- `requirements.txt` — partial Python dependencies.
- `Dockerfile` — an initial container definition; it is not currently runnable because it expects `main.py`.
- `Video Moment Retrieval — Project Formation.md` — product goals, proposed architecture, evaluation plan, and references.

## Environment setup

The notebook uses Python, Jupyter, FFmpeg, Whisper, PyTorch, pandas, NumPy, and Sentence Transformers. Install FFmpeg first (for example, on macOS):

```bash
brew install ffmpeg
```

Create and activate a virtual environment, then install the project requirements and the notebook's currently missing packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter torch sentence-transformers
```

The notebook loads the Whisper `small` model and `google/embeddinggemma-2`; the first run may download model weights. Embedding can take a long time, depending on the video set and available hardware. The notebook selects Apple's MPS device when available and otherwise uses CPU.

## Notebook workflow

Open `notebooks/MM_embedding.ipynb` in Jupyter with the project environment selected as its kernel. The notebook uses paths under `notebooks/note_assets/`, so start Jupyter from the `notebooks` directory:

```bash
cd notebooks
jupyter notebook MM_embedding.ipynb
```

The notebook includes cells that create the asset directories, split input media with FFmpeg, optionally transcribe audio with Whisper, and embed video clips. It may take substantial time to process all clips. If a notebook run is already in progress, leave that kernel running; do not start the embedding cells in a second kernel against the same files.

Segment CSV files record each generated clip's filename and start/end times. The current embedding experiment builds an embedding matrix in memory and does not save a reusable index or expose query search. Keep generated clips and CSV files if you need to continue working with that notebook output.

## Planned direction

The project plan proposes a multimodal retrieval pipeline: index speech and visual segments once, embed a text query, retrieve and rank matching moments, then consider a video LLM reranker for precise temporal grounding. That full flow, its evaluation metrics, and the adapter training approach have not yet been implemented.

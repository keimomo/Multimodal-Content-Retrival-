# Video Moment Retrieval — Project Formation

Oct 15, 2026 · @k

## Overview

We are building a system that takes a natural-language text query and returns the specific moments in a video library that match it, as a ranked list of time ranges with thumbnails.

Today, finding a moment in long video means scrubbing manually or relying on chapter titles. Users want to ask things like "where does the speaker explain the loss function" or "the part where the dog jumps into the pool" and jump straight there.

**Input:** a text query, optionally scoped to one video or a collection.

**Output:** a ranked list of candidate moments, each with `video_id`, `start`, `end`, a thumbnail, and the matching transcript snippet where there is one.

We are exploring two approaches in parallel: **(A)** a multimodal embedding retrieval pipeline, and **(B)** an adapter-based video LLM that grounds the query to timestamps. The expected end state combines them: A narrows the search, B pinpoints the moment.

## High-level architecture

The system has two halves: an offline indexing path that runs once per video, and an online query path that runs per search and only reads the indexes.

[embedded content: high-level architecture · offline indexing and online query

The upper half is Approach A's indexing. The lower half is A's query path, with the dashed box marking where Approach B plugs in: first as a generic VLM reranker, later as our adapter-based video LLM.

## Goals and success metrics

The project succeeds if the correct moment appears in the top 5 results for most real user queries, fast enough to feel interactive.

**Goals**

- Retrieve moments matched on what is **said** (speech) and what is **shown** (visuals).
- Handle simple queries ("red car parks") and temporal ones ("after the coffee break, when he shows the formula").
- Scale to a library of many hours of video without re-processing per query.
- Compare A and B on the same test set and decide how to combine them.

**Non-goals (for now)**

- Generating answers or summaries from the video (full RAG). We return moments only.
- Live or streaming video.
- Training a vision encoder or LLM from scratch.

**Metrics**


| Metric              | What it measures                              | Target                 |
| ------------------- | --------------------------------------------- | ---------------------- |
| Recall@5            | Correct moment is in the top 5                | To set after baseline  |
| Recall@1            | Correct moment is ranked first                | To set after baseline  |
| mIoU / R@1 IoU≥0.5  | Overlap between predicted and true time range | To set after baseline  |
| Query latency (p95) | Time from query to results                    | Interactive, under 2 s |
| Indexing cost       | GPU time per hour of video                    | Tracked, no target yet |




## Approach A: multimodal embedding retrieval

Every video is cut into timestamped segments, each segment is embedded once, and a query is answered by nearest-neighbour search over those vectors. No model reasons at query time; it is fast, cheap and scales to large libraries.

The embedding model is [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (Google, Apache 2.0), which maps text, images, video and audio into one shared 768-dimensional space, so a text query can be compared directly with frames and transcript chunks.

**Indexing tracks (offline, once per video)**

1. **Transcript track.** Speech-to-text (e.g. Whisper) with word timestamps. Words are grouped into sentence-based windows of about 10–30 s with small overlap, and each window's text is embedded.
2. **Visual track.** Frames at 1 fps, or 2–4 s clips, ideally cut at shot boundaries (e.g. PySceneDetect). Each segment is embedded as image or video. This catches moments that are never spoken.
3. **Caption track (optional).** A vision-language model writes a one-line description per visual segment, which is embedded as text. Slower to index but often more precise for detailed queries.

Each vector is stored with `{video_id, start, end, track}` in a vector index (FAISS, Qdrant or pgvector).

**Query path (online)**

1. Embed the query with the retrieval task prefix.
2. Search each track's index for the top-k segments.
3. Fuse the ranked lists with Reciprocal Rank Fusion.
4. Merge adjacent hits into a single time range.
5. Return the ranked moments.

**Strengths:** milliseconds per query, no training needed to start, scales to thousands of hours.

**Weaknesses:** weak on temporal logic ("after X", "the second time"), precision limited by segment size, generic embeddings may miss domain-specific meaning.

**Possible upgrade:** a small LoRA or linear adapter trained on our own query–moment pairs to specialize the embeddings to our content.

## Approach B: adapter + video LLM (temporal grounding)

An LLM "watches" a clip through a trained adapter and generates the start and end time of the moment the query describes. It reasons over the query and the video together, so it handles complex and temporal queries that similarity search cannot.

[embedded content: Approach B architecture · visual path, text path, LLM

Frames go through the frozen encoder and the trained adapter; the query and transcript go through the normal tokenizer. The LLM reads both token streams together and outputs the moment's start and end.

**Components**

1. **Vision encoder (frozen).** Turns each sampled frame into feature vectors.
2. **Adapter / projector (trained).** A small network (MLP or Q-Former-style) that maps visual features into the LLM's token space, so frames read like tokens to the LLM.
3. **Timestamp encoding.** Frame tokens are interleaved with their times (e.g. `t=12s: [frame tokens]`) so the model can refer to positions in the video.
4. **LLM (frozen or LoRA-tuned).** Reads the query plus frame and transcript tokens and outputs e.g. `41.0 – 47.5 s`.

This follows the LLaVA-style design used by temporal-grounding research models such as TimeChat and VTimeLLM.

**Training**

- Stage 1: train the adapter on video–caption pairs so visual tokens align with language.
- Stage 2: fine-tune adapter + LoRA on (video, query, start, end) examples for grounding.
- Data: public sets (QVHighlights, Charades-STA, ActivityNet Captions) plus a few hundred to a few thousand labelled examples from our own domain.

**Strengths:** handles temporal and multi-step queries, can combine speech and visuals in one judgement, can explain or verify its choice.

**Weaknesses:** seconds and a GPU per query, context window limits how much video it sees at once (long videos must be chunked or subsampled), timestamp numbers can be imprecise, needs labelled training data.

## Comparison and combined design

The two approaches are complementary: A is fast and scales but reasons poorly; B reasons well but is slow and limited to short spans of video. The target design runs A first to shortlist candidates and B second to pick and refine the moment.


|                            | A: Embedding retrieval                     | B: Adapter + video LLM                      |
| -------------------------- | ------------------------------------------ | ------------------------------------------- |
| How it finds the moment    | Similarity search over precomputed vectors | Generates timestamps after reading the clip |
| Query latency              | Milliseconds                               | Seconds, GPU per query                      |
| Video length               | Whole library, indexed once                | Limited by context window                   |
| Temporal / complex queries | Weak                                       | Strong                                      |
| Timestamp precision        | Bounded by segment size                    | Fine-grained but sometimes off              |
| Training needed            | None to start                              | Adapter + LoRA, labelled data               |
| Main risk                  | Misses meaning not captured by embeddings  | Cost, hallucinated timestamps               |


**Combined flow:** A retrieves the top 20 candidate moments across the library; each candidate is padded by a few seconds and passed, with its transcript snippet, to B; B confirms or rejects each candidate and tightens its start and end; results are re-ranked by B's confidence.

[embedded content: combined architecture · stage A then stage B

Stage A searches every indexed video in milliseconds; stage B only sees the 20 clips A hands it, which keeps GPU cost bounded.

## Plan, evaluation and risks

We build A first because it needs no training and gives a working baseline; B is developed against the same test set and plugged in as the second stage.

**Phases**

1. **Test set.** Collect 50–100 real queries on our videos, each labelled with the correct start and end.
2. **A baseline.** Transcript + visual tracks, RRF fusion, merging. Measure Recall@1/5.
3. **A improvements.** Add caption track, shot-based segmentation, tune window sizes. Keep only what moves the metrics.
4. **B prototype.** Fine-tune adapter + LoRA on public grounding data, evaluate on the shortlisted candidates from A.
5. **B on our domain.** Label our own (query, moment) examples and fine-tune further.
6. **Combined system.** A → B pipeline, measure accuracy against latency and cost.

**Evaluation**

- Same test set for every variant, split by query type: spoken, visual-only, temporal/complex.
- Report Recall@1, Recall@5, R@1 IoU≥0.5 and p95 latency for each.
- Read failure cases by hand after every run; most fixes come from there.

**Risks**

- **Visual-only moments** are missed if we rely on transcripts → visual track is mandatory from day one.
- **Labelled data** for B is slow to produce → start with public datasets, label incrementally.
- **GPU cost** of B at query time → only run it on A's shortlist, cap candidates.
- **Imprecise timestamps from B** → constrain output to the candidate window and fall back to A's boundaries.

**Open questions**

- [ ] What content will we index first (lectures, meetings, sports, vlogs)? This decides whether speech or visuals dominate.
- [ ] How large is the initial library, in hours?
- [ ] Which base video LLM do we start B from?
- [ ] On-device or server deployment?

**Sources**

- [EmbeddingGemma 2 model card (Hugging Face)](https://huggingface.co/google/embeddinggemma-2)
- [EmbeddingGemma 2 serving guide (SGLang)](https://docs.sglang.io/cookbook/autoregressive/Google/EmbeddingGemma2.md)


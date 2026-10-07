# arXiv RAG Explorer

**Grounded, cited answers over a collection of research abstracts, running 100% locally.**
LangGraph orchestrates a LangChain pipeline: query understanding with **Qwen 3.5** (via Ollama), hybrid retrieval
(**Chroma** vectors + **SQLite FTS5** BM25), local **cross-encoder reranking**, and streamed generation, with an
optional illustration from an external image API.

![Answer view: pipeline progress, streamed answer with citation chips, retrieved context with rank badges](docs/screenshot-light.png)

<table><tr>
<td><img src="docs/screenshot-dark.png" alt="Dark mode" width="560"></td>
<td><img src="docs/screenshot-mobile.png" alt="Mobile layout" width="200"></td>
</tr></table>

---

## Contents

- [Quick start](#quick-start)
- [Start, stop and restart](#start-stop-and-restart)
- [Assignment checklist](#assignment-checklist)
- [Architecture](#architecture)
- [Using another dataset](#using-another-dataset)
- [API](#api)
- [Bonus: image generation](#bonus-image-generation)
- [Configuration](#configuration)
- [Models and hardware](#models-and-hardware)
- [Retrieval quality benchmark](#retrieval-quality-benchmark)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Design decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)

---

## Quick start

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com/download) (it serves Qwen locally). No Docker.

```bash
git clone <your-repo-url> arxiv-rag && cd arxiv-rag
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                                          # → http://127.0.0.1:8080
```

That's all. On the first start the app:

1. waits for Ollama (the page tells you if it isn't running),
2. **downloads the models automatically**: `qwen3.5:4b` (3.3 GB) and `nomic-embed-text` (274 MB) through Ollama,
   plus the 90 MB reranker from Hugging Face,
3. streams `data/arxiv_2.9k.jsonl` into the local index (~1.5 min on an Apple M1 Pro; reused on later starts),
4. serves the UI at **http://127.0.0.1:8080** and the API docs at **/docs**.

The browser shows live progress for each of these steps.

**Prefer a single command?** The setup scripts create the venv, install the requirements, install/start Ollama if
needed and download every model up front:

```bash
./scripts/setup.sh                                         # macOS / Linux
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Windows
```

You can also pre-download the models on their own with `python -m rag_app.setup_models`
(for example `--llm qwen3.5:9b`).

> **Why aren't the models in `requirements.txt`?** pip can only install Python packages. Model weights are
> downloaded by the `ollama` client (listed in `requirements.txt`) the first time the app starts, so
> `pip install -r requirements.txt` + `python main.py` is the whole minimal installation.

---

## Start, stop and restart

The index, downloaded models and settings persist on disk, so stopping is safe and a restart takes only a few
seconds (no re-indexing, no re-downloading). Run the commands from the project folder.

| Action | macOS / Linux | Windows (PowerShell) |
|---|---|---|
| Start | `source .venv/bin/activate && python main.py` | `.venv\Scripts\python.exe main.py` |
| Stop it (started in a terminal) | <kbd>Ctrl</kbd>+<kbd>C</kbd> in that terminal | <kbd>Ctrl</kbd>+<kbd>C</kbd> |
| Start in the background | `nohup .venv/bin/python main.py > rag.log 2>&1 &` | `Start-Process .venv\Scripts\python.exe main.py -WindowStyle Hidden` |
| Stop it (running in the background) | `kill $(lsof -tiTCP:8080 -sTCP:LISTEN)` | `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8080 -State Listen).OwningProcess` |
| Is it running? | `curl -s http://127.0.0.1:8080/health` | `Invoke-RestMethod http://127.0.0.1:8080/health` |

If you started the app with another `PORT`, use that number instead of `8080` in the stop and health commands.

**PyCharm:** select the project's `.venv` as the interpreter, open `main.py` and press ▶ *Run*; the ■ *Stop*
button stops it.

**Ollama runs separately.** Stopping the app doesn't stop Ollama. Idle models are unloaded automatically after 30
minutes (`LLM_KEEP_ALIVE`); to free the memory right away run `ollama stop qwen3.5:4b`, or quit the Ollama app.

---

## Assignment checklist

Every requirement of the home assignment, and where it is met. Docker packaging was deliberately left out: the
project runs natively with one command and is published as a buildable repository.

| Requirement | How it is met |
|---|---|
| Load a `.jsonl` of abstracts without assuming it fits in memory | `JsonlAbstractLoader` (a LangChain `BaseLoader`) streams line by line; indexing works in batches of 64; hashing and line counting are streamed too. Only `id`, `title` and `abstract` are kept. |
| Index locally | Dense vectors in **Chroma** (persistent, cosine HNSW) + BM25 keyword index in **SQLite FTS5**, under `storage/index/`. |
| Read `DATA_PATH` on startup, build or load the index | The index is fingerprinted (SHA-256 of the file + embedding model + chunking settings). A matching index loads in ~1 s; otherwise it is built. |
| New dataset → discard the old index and rebuild | A changed fingerprint drops the old index from service, builds a new one and deletes the old vectors and keywords. It works on restart, on a live file change (watcher), from the UI upload, and via `POST /api/reindex`. |
| Understand the query | LangGraph node `understand_query`: Qwen returns a JSON-schema-constrained plan (search rewrites, keywords, topic). It also translates non-English questions. |
| Retrieve relevant entries | Weighted Reciprocal Rank Fusion of dense + BM25 lists, then a local cross-encoder reranks 20 candidates down to `top_k`. |
| Coherent answer grounded in the context | Qwen answers only from the numbered abstracts, cites inline as `[n]`, and says when the context is insufficient instead of inventing papers or facts. |
| Citations (doc id + title) and retrieved context in the response | `citations: [{doc_id, title, ref, url}]` and `retrieved_context: ["..."]`, exactly as in the spec, plus `sources`, `query_analysis` and `metrics`. |
| Web UI at `http://127.0.0.1:8080` with input box, answer, citations, retrieved context | Custom UI, also offline (no CDNs): live pipeline steps, streamed answer with clickable citation chips, context cards, raw JSON, history, dataset panel, dark mode, mobile layout. |
| Bonus: `/answer` and `/stream` endpoints | `POST/GET /answer` (JSON) and `POST/GET /stream` (Server-Sent Events). OpenAPI docs at `/docs`. |
| Fully on-prem indexing, retrieval and generation | Ollama (Qwen + embeddings), Chroma, SQLite and ONNX Runtime all run locally. Only the optional image generation may call a (free) external API. |
| Bonus: optional image generation, UI toggle, in the JSON, must not slow the text | "Generate image" switch → `images: [...]` in the JSON. Free providers only: a local Ollama model first, then the free Cloudflare Workers AI and Pollinations APIs. External APIs run as a **parallel LangGraph branch**; the local model runs after the answer. Tests assert that the text is never delayed. |
| Runs with a single command | `python main.py` (or `DATA_PATH=/path/to/file.jsonl python main.py`). |
| Works with any `.jsonl` dataset | Field aliases (`text`/`content`/`summary`…, `doc_id`/`paper_id`…), malformed lines and duplicate ids are skipped, and long texts are chunked. A directory or URL also works as `DATA_PATH`. |
| CPU-only, 4–8 GB RAM | Ollama runs on CPU or GPU, the reranker on CPU, and model size is configurable (see [hardware](#models-and-hardware)). |
| README incl. image-generation setup | This file, see [Bonus: image generation](#bonus-image-generation). |

---

## Architecture

### Query pipeline (LangGraph)

```mermaid
flowchart LR
    Q([question]) --> U["understand_query<br/><small>Qwen · JSON-schema plan</small>"]
    U --> R["retrieve<br/><small>Chroma + FTS5 BM25<br/>weighted RRF · 20 candidates</small>"]
    R --> X["rerank<br/><small>MiniLM cross-encoder<br/>ONNX · top-k</small>"]
    X --> G["generate_answer<br/><small>Qwen · streamed · cited</small>"]
    X -. "generate_image = true" .-> I["generate_image<br/><small>external API</small>"]
    G --> F["finalize<br/><small>response JSON</small>"]
    I --> F
```

| Node | What it does |
|---|---|
| `understand_query` | Qwen (`format=json_schema`, thinking off) turns the question into 1–2 English search rewrites, keywords and a topic. Guesses are contained: BM25 only gets keywords whose words occur in the question, and an unanchored topic falls back to the question's own terms. On any failure a heuristic plan takes over. |
| `retrieve` | Embeds all queries in one batch, runs one dense search per query plus one BM25 search, and fuses everything with **weighted RRF** (`Σ w / (60 + rank)`), keeping the best chunk per paper. |
| `rerank` | A local cross-encoder reads (question, abstract) pairs together. Its ranking is blended with the hybrid ranking (RRF again) to pick the final `top_k`. If the model is unavailable, the hybrid order passes through. |
| `generate_answer` | Streams tokens from Qwen. Citations are parsed from `[n]` markers and repaired when the model writes `[Abstract 2]` or a raw doc id. |
| `generate_image` | Runs in the **same super-step** as `generate_answer` (LangGraph executes them concurrently), so the image never delays the text. |
| `finalize` | Builds the response: `answer`, `citations`, `retrieved_context`, `sources`, `query_analysis`, `images`, `metrics`. |

Each node publishes progress through LangGraph's custom stream (`get_stream_writer`). `/stream` forwards these
events as Server-Sent Events, and they drive the UI's pipeline indicator.

### Indexing

```mermaid
flowchart LR
    D[("DATA_PATH<br/>.jsonl")] -->|stream lines| L[JsonlAbstractLoader]
    L -->|"dedupe ids, split texts > 2000 chars"| B[batches of 64]
    B --> E["Ollama embeddings<br/>nomic-embed-text"]
    E --> C[("Chroma<br/>cosine HNSW")]
    B --> K[("SQLite FTS5<br/>BM25")]
    C --> M["manifest.json<br/>(written last = commit)"]
    K --> M
```

- **Fingerprint** = SHA-256(dataset) + embedding model + document prefix + chunk size/overlap.
  The same content under another name reuses the index; any real change triggers a rebuild.
- **Crash-safe:** each build gets fresh, uniquely named collection and keyword files, and `manifest.json` is
  replaced atomically only after success. An interrupted build is simply rebuilt next time.
- **Model-specific prefixes:** e.g. nomic's `search_document:` / `search_query:`, applied automatically.
- **Live progress** (`/api/status`): processed / total, docs/s and ETA, shown in the UI as a progress bar.

---

## Using another dataset

```bash
DATA_PATH=/path/to/arxiv_5k.jsonl python main.py     # any .jsonl file
python main.py --data-path /data                     # a directory: its newest .jsonl is used
DATA_PATH=https://example.org/papers.jsonl python main.py   # downloaded (streamed) and cached
```

Each line should look like `{"id": "...", "title": "...", "abstract": "..."}`. Other fields are ignored, and common
aliases are accepted. On start the old index is discarded and a new one is built automatically. While the server
is running you can also:

- **replace the file in place**: the watcher notices within ~20 s and rebuilds (`WATCH_DATASET=true`),
- **upload a file** from the UI (*Dataset → drop a .jsonl*) or with `POST /api/dataset`,
- **force a rebuild** with *Dataset → Rebuild index* or `POST /api/reindex`.

> **Docker-command equivalent.** The assignment's example sets `DATA_PATH=/data/arxiv_5k.jsonl` but mounts
> `arxiv_2.9k.jsonl`. If `DATA_PATH` names a file that doesn't exist, the app uses the `.jsonl` next to it and logs
> a warning instead of failing.

---

## API

Interactive docs: **http://127.0.0.1:8080/docs**.

### `POST /answer` (JSON)

```bash
curl -s -X POST http://127.0.0.1:8080/answer \
  -H 'Content-Type: application/json' \
  -d '{"query": "What techniques make diffusion models generate images in only a few steps?", "top_k": 5, "generate_image": false}'
```

Also `GET /answer?query=...&top_k=5&generate_image=false`. The body accepts `query` (aliases `question`, `q`),
`top_k` (1–20) and `generate_image`.

<details><summary>Example response (real output, trimmed with "...")</summary>

```json
{
  "answer": "Several techniques described in the abstracts enable diffusion models to generate images or text in significantly fewer sampling steps, primarily by altering the sampling trajectory or model architecture.\n\n* **Penalizing Boundary Activation:** ... during early denoising steps [1]. ...\n* **Rectified Flow:** The RFfusion approach incorporates Rectified Flow to straighten the sampling path, enabling one-step sampling for image fusion tasks [5]. ...\n* **Few-Step Discrete Flow-Matching (FS-DFM):** ... trains the model to be consistent across different step budgets [4]. ...\n* **Cascaded Low-Resolution Generation:** LowDiff ... generates images progressively from low resolution to high resolution using a unified model [3]. ...",
  "citations": [
    {"ref": 1, "doc_id": "2509.16968v2", "title": "Penalizing Boundary Activation for Object Completeness in Diffusion Models", "url": "https://arxiv.org/abs/2509.16968v2"},
    {"ref": 5, "doc_id": "2509.16549v2", "title": "Efficient Rectified Flow for Image Fusion", "url": "https://arxiv.org/abs/2509.16549v2"},
    {"ref": 4, "doc_id": "2509.20624v1", "title": "FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Models", "url": "https://arxiv.org/abs/2509.20624v1"},
    {"ref": 3, "doc_id": "2509.15342v1", "title": "LowDiff: Efficient Diffusion Sampling with Low-Resolution Condition", "url": "https://arxiv.org/abs/2509.15342v1"}
  ],
  "retrieved_context": ["Diffusion models have emerged as a powerful technique for text-to-image (T2I) generation, creating high-quality ...", "..."],
  "sources": [
    {"ref": 1, "doc_id": "2509.16968v2", "chunk_id": "2509.16968v2#0", "title": "Penalizing Boundary Activation for Object Completeness in Diffusion Models",
     "text": "...", "score": 0.04817, "similarity": 0.8329, "dense_rank": 1, "keyword_rank": 11, "hybrid_rank": 5,
     "rerank_rank": 1, "rerank_score": 4.137, "url": "https://arxiv.org/abs/2509.16968v2"},
    "..."
  ],
  "query_analysis": {
    "search_queries": ["accelerated diffusion sampling methods few-step image generation", "distillation techniques for fast diffusion model inference"],
    "keywords": ["diffusion models", "few-shot sampling", "distillation", "inference speed", "latent diffusion"],
    "topic": "accelerated diffusion image generation",
    "queries_used": ["What techniques make diffusion models generate images in only a few steps?", "accelerated diffusion sampling methods few-step image generation", "distillation techniques for fast diffusion model inference"],
    "query_weights": [1.0, 1.0, 1.0],
    "bm25_keywords": ["diffusion models"],
    "method": "llm"
  },
  "images": [],
  "image_error": null,
  "metrics": {
    "latency_ms": 14486.1, "total_ms": 14487.9, "analysis_ms": 4253.5, "retrieval_ms": 180.6, "rerank_ms": 664.1,
    "generation_ms": 9383.2, "time_to_first_token_ms": 444.4, "image_ms": null, "prompt_tokens": 1644,
    "completion_tokens": 314, "tokens_per_second": 35.1, "model": "qwen3.5:4b", "embedding_model": "nomic-embed-text",
    "top_k": 5, "candidates": 20,
    "retrieval_mode": "hybrid: dense (Chroma) + BM25 (SQLite FTS5), Reciprocal Rank Fusion + cross-encoder rerank (cross-encoder/ms-marco-MiniLM-L6-v2)"
  }
}
```

Note how the plan's guessed keywords ("few-shot sampling", "latent diffusion") were *not* sent to BM25, and how the
reranker promoted a paper from hybrid rank 5 to rank 1.
</details>

### `POST /stream` (Server-Sent Events)

```bash
curl -N -X POST http://127.0.0.1:8080/stream -H 'Content-Type: application/json' -d '{"query": "What is RLBFF?"}'
curl -N "http://127.0.0.1:8080/stream?query=What%20is%20RLBFF%3F"     # GET works with EventSource too
```

| Event | Payload |
|---|---|
| `step` | `{step, status: running/done, ms?}` for `understand`, `retrieve`, `rerank`, `generate`, `image` |
| `analysis` | the query plan (rewrites with their RRF weights, keywords, topic) |
| `retrieval` | the final passages, after reranking |
| `token` | `{text}`: answer tokens as they are generated |
| `thinking` | reasoning tokens (only with `LLM_THINKING=true`) |
| `answer` | `{answer, citations}`: the text is complete (the image may still be generating) |
| `image` | `{images, error, prompt, ms}` |
| `final` | the full response, identical to `/answer` |
| `error`, `done` | failure details / end of stream |

### Operations

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | lifecycle state, download/indexing progress, dataset & index details, models |
| `GET /health` | `200 {"status":"ok"}` when ready, otherwise `503` |
| `POST /api/reindex` | discard and rebuild the index |
| `POST /api/dataset` | upload a new `.jsonl` (multipart), which then replaces the index |
| `GET /api/examples` | random paper titles (example questions for any dataset) |

---

## Bonus: image generation

Tick **Generate image** in the UI (or send `"generate_image": true`). Only **free** options are used, tried in this
order, and if one fails the next one is used automatically:

| # | Provider | Cost | Setup |
|---|---|---|---|
| 1 | **Local model via Ollama** (`x/flux2-klein:4b`, FLUX.2 [klein], Apache-2.0) | free, offline | `ollama pull x/flux2-klein:4b` (5.7 GB), or `python -m rag_app.setup_models --image` |
| 2 | **Cloudflare Workers AI** (FLUX.1 schnell) | free plan: 10,000 neurons/day ≈ 170 images/day | free Cloudflare account → *AI → Workers AI* → create an API token with the *Workers AI* permission, then set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` |
| 3 | **Pollinations.ai** (FLUX) | free | nothing (keyless, rate-limited and watermarked); a free account key in `POLLINATIONS_TOKEN` removes the watermark |

How it is wired:

- **The text is never slowed down.** External APIs run in a parallel LangGraph branch next to the answer. The local
  model shares the GPU with Qwen, so it starts *after* the answer is complete (the UI step says "after answer").
- **The JSON response** contains `images: [{url, prompt, provider, model, latency_ms}]`; files are served from
  `/generated/`. If every provider fails, `image_error` explains why, and the answer is unaffected.
- **No extra LLM call:** the image prompt is built deterministically from the query plan (topic + keywords).
- **Force one provider** with `IMAGE_PROVIDER=ollama|cloudflare|pollinations`, or disable the feature with
  `IMAGE_PROVIDER=none`. Models: `OLLAMA_IMAGE_MODEL`, `CLOUDFLARE_IMAGE_MODEL`, `POLLINATIONS_MODEL`.

> **Local generation status (October 2026):** Ollama introduced local image generation in January 2026, but
> **temporarily removed it in version 0.32.6** (it answers *"image generation models are not currently
> supported"*). The app detects this on the first request and falls back to the free APIs. Once Ollama restores
> the feature, local generation switches on again after an upgrade, with no code change. Note that FLUX.2 klein
> 4B needs a lot of memory; on 16 GB machines Ollama may refuse it, and the fallback handles that too.

**Why no Gemini, OpenAI or Hugging Face?** They aren't free for image generation through their APIs (checked
October 2026). Gemini's "Nano Banana" models have no free API tier: they're free only inside the Gemini / AI Studio
web apps, which a program can't call. OpenAI image models require paid credits, and free Hugging Face accounts get no
Inference Providers credits. To keep the project free, only free providers are built in.

---

## Configuration

All settings are environment variables (or `.env`). See [`.env.example`](.env.example) for the full,
commented list. The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `data/arxiv_2.9k.jsonl` | dataset file, directory or URL |
| `HOST` / `PORT` | `127.0.0.1` / `8080` | web server |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | also honours `OLLAMA_HOST` |
| `LLM_MODEL` | `qwen3.5:4b` | any Ollama chat model |
| `EMBED_MODEL` | `nomic-embed-text` | any Ollama embedding model (`bge-m3` for multilingual) |
| `AUTO_PULL_MODELS` | `true` | download missing models on startup |
| `TOP_K` | `5` | abstracts given to the LLM (UI slider: 1–10, API: 1–20) |
| `QUERY_REWRITE` | `true` | LLM query understanding (else a keyword heuristic) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | `none` disables reranking |
| `REWRITE_WEIGHT` / `KEYWORD_WEIGHT` | `auto` | RRF weights (1.0/1.0 with the reranker, 0.25/1.5 without) |
| `LLM_THINKING` | `false` | let Qwen think first; the reasoning streams into a collapsible panel |
| `STORAGE_DIR` | `storage` | index, models, uploads, images |
| `IMAGE_PROVIDER` | `auto` | `auto` (local → Cloudflare → Pollinations), `ollama`, `cloudflare`, `pollinations` or `none`; see [image generation](#bonus-image-generation) |

CLI flags override the environment: `python main.py --data-path ... --port 9000 --llm qwen3.5:9b`.

---

## Models and hardware

| Component | Default | Download | Runs on |
|---|---|---|---|
| LLM | `qwen3.5:4b` | 3.3 GB | Ollama (Metal / CUDA / CPU) |
| Embeddings | `nomic-embed-text` (768-d) | 274 MB | Ollama |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | 90 MB | ONNX Runtime, CPU |

| RAM | Suggested `LLM_MODEL` |
|---|---|
| 4 GB | `qwen3.5:0.8b` (1.3 GB) or `qwen3:1.7b` (1.4 GB) |
| 8 GB (recommended) | `qwen3.5:4b` (default) |
| 16 GB+ | `qwen3.5:9b` (6.6 GB) for richer answers |

Measured on an Apple M1 Pro (16 GB): indexing 2,900 abstracts takes ~83 s (~35 docs/s); a later start loads the
index in ~1 s. A typical question costs ~4 s of planning, ~0.2–0.9 s of retrieval, ~0.6 s of reranking and the
first answer token ~3–5 s later. Generation then runs at ~35 tokens/s, so a full answer takes ~12–25 s. On a
CPU-only machine everything works, just slower. Pick a smaller model, or set `QUERY_REWRITE=false` to skip the
planning call.

---

## Retrieval quality benchmark

`scripts/eval_retrieval.py` measures whether the paper that answers a question is retrieved. It needs no labelled
data: the local LLM writes the questions from random abstracts, and each question has exactly one relevant paper.

- **semantic**: natural search-engine questions (n=40)
- **paraphrased**: the same papers, but the question must avoid the abstract's terminology (n=40). This
  vocabulary mismatch is closest to how non-experts ask.
- **named**: "What is RLBFF?"-style lookups of papers whose title starts with a name/acronym (n=20)

Results with k=5 on the bundled dataset (`qwen3.5:4b`, `nomic-embed-text`):

| Strategy | semantic hit@5 | paraphrased hit@5 | paraphrased MRR@5 | named hit@1 | named hit@5 |
|---|---|---|---|---|---|
| dense only | 0.97 | 0.57 | 0.45 | 0.60 | 0.80 |
| BM25 only | 1.00 | 0.45 | 0.37 | 0.95 | 0.95 |
| hybrid (RRF) | 1.00 | 0.60 | 0.46 | 0.80 | 0.95 |
| hybrid + LLM query plan | 1.00 | 0.62 | 0.47 | 0.95 | 1.00 |
| **hybrid + plan + rerank (default)** | **1.00** | **0.80** | **0.59** | **0.95** | **1.00** |

Reranking adds ~0.57 s per question. The benchmark drove three design decisions:

1. **Weighted fusion.** Giving LLM rewrites full weight without a reranker let a drifting rewrite (e.g. a guessed
   acronym expansion) outvote the question.
2. **Question-anchored BM25 keywords.** Planner keywords such as "latent diffusion" for a question about
   "diffusion" were pulling exact matching off-topic.
3. **The cross-encoder.** It's the only configuration that is best or tied-best on every query set.

Run it yourself (stop the web app first so the two don't compete for the GPU):
`python scripts/eval_retrieval.py --n 40 --named 20 --k 5`.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # 47 tests, ~3 s, fully offline
```

The suite never touches Ollama or the network: it uses deterministic bag-of-words embeddings and LangChain's
`GenericFakeChatModel`. It covers:

- streaming loading, malformed and duplicate records, `DATA_PATH` fallbacks;
- fingerprint reuse vs rebuild, cleanup of old indexes, forced rebuilds;
- hybrid retrieval and its ablation modes, weighted fusion, reranker blending;
- citation parsing;
- the API schema, the SSE event order, validation errors, `503` before the index exists, and dataset upload →
  rebuild;
- the image branch: success, failure, and the guarantee that a slow image doesn't delay the answer;
- the free image chain (local Ollama, Cloudflare Workers AI, Pollinations) against mocked HTTP: fallbacks,
  error messages, and that a local model runs after the answer instead of competing with it.

---

## Project layout

```
├── main.py                  # python main.py → http://127.0.0.1:8080
├── requirements.txt         # runtime deps (models are pulled automatically, see Quick start)
├── requirements-dev.txt     # + pytest
├── .env.example             # every setting, documented
├── data/arxiv_2.9k.jsonl    # sample dataset (arXiv metadata)
├── rag_app/
│   ├── config.py            # settings from env / .env
│   ├── dataset.py           # DATA_PATH resolution, streaming LangChain loader, fingerprinting
│   ├── embeddings.py        # Ollama embeddings + model-specific task prefixes
│   ├── keyword_index.py     # SQLite FTS5 BM25 index
│   ├── index_manager.py     # build / load / swap / discard the index (Chroma + FTS5)
│   ├── retriever.py         # HybridRetriever (LangChain BaseRetriever), weighted RRF
│   ├── reranker.py          # ONNX cross-encoder + rank blending
│   ├── llm.py               # ChatOllama factories, structured query plan
│   ├── prompts.py           # analysis and answer prompts
│   ├── graph.py             # LangGraph pipeline (nodes, fan-out, streaming events)
│   ├── images.py            # bonus image providers
│   ├── service.py           # startup orchestration, dataset watcher, answer/stream
│   ├── server.py            # FastAPI: UI, /answer, /stream, /api/*
│   ├── setup_models.py      # Ollama checks + model downloads (also a CLI)
│   └── static/              # the web UI (vanilla HTML/CSS/JS, no build step)
├── scripts/
│   ├── setup.sh / setup.ps1 # one-shot installers
│   └── eval_retrieval.py    # retrieval benchmark
└── tests/                   # offline pytest suite
```

---

## Design decisions

- **Ollama for every model call.** One local runtime serves the LLM and the embeddings, with GPU acceleration
  where available and CPU otherwise. Model downloads are a single API call, which is what makes
  "pip install + run" possible without Docker.
- **Qwen 3.5 4B with thinking off.** It is fast (~35 tok/s on an M1) and its structured-output support makes query
  planning reliable. Thinking can be switched on (`LLM_THINKING=true`): the reasoning streams into a collapsible
  panel and an extra token budget is reserved for it. Expect answers to take ~4× longer (≈60 s instead of ≈15 s).
- **`nomic-embed-text` over `bge-m3`.** It indexes ~3× faster (≈83 s vs ≈4 min for this dataset) with equal
  results on our probes. Indexing speed matters because every new dataset triggers a rebuild, and on CPU-only
  machines it decides whether startup takes minutes or an hour. Cross-lingual questions still work: the planner
  translates them into English queries (ask in Hebrew, get a Hebrew answer grounded in English abstracts).
- **Chroma + SQLite FTS5 instead of one hybrid engine.** Chroma's hybrid search API is cloud-only. FTS5 ships with
  Python, lives on disk (so the dataset never has to fit in memory) and gives proper BM25. Fusion is a few lines of
  RRF.
- **Whole abstracts as chunks.** Every abstract fits easily in the embedding context, and one chunk per paper keeps
  citations clean. Longer texts (other datasets) are split with LangChain's `RecursiveCharacterTextSplitter`.
- **Blue/green-style index lifecycle.** Every build writes uniquely named collection and keyword files and commits
  by atomically replacing `manifest.json`, so a crash or a forced rebuild never corrupts or reuses the files of the
  index being served (a bug the test suite caught and now guards against).
- **Streaming everywhere.** Tokens, pipeline steps and partial results stream to the UI, so the user sees progress
  within a second even though a local 4B model needs ~15 s for a full answer.
- **No Docker (by choice).** The app runs natively (`python main.py`) and is published as a buildable repository.
  Everything a container would provide is still here: one command, local models, a configurable `DATA_PATH` and
  port `8080`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says *Waiting for Ollama* | Start it (open the Ollama app or run `ollama serve`), or set `OLLAMA_BASE_URL`. |
| First start is slow | Models are downloading; progress is shown in the UI and the terminal. Later starts take seconds. |
| The first answer after a while is slow | Ollama unloads idle models after `LLM_KEEP_ALIVE` (30 min). Other apps using *different* Ollama models at the same time can also force model swaps. |
| Out of memory / very slow on CPU | Use a smaller `LLM_MODEL` (see [hardware](#models-and-hardware)), lower `LLM_NUM_CTX` to `4096`, or set `QUERY_REWRITE=false`. |
| `Reranker unavailable` in the logs | The first run couldn't reach Hugging Face. The app keeps working without reranking; it retries on the next start. Disable it with `RERANKER_MODEL=none`. |
| Image: *keyless tier is rate-limited / failed* | The keyless tier is best-effort (rate limits, occasional outages). Add a free `POLLINATIONS_TOKEN` or a free Cloudflare Workers AI account (see [image generation](#bonus-image-generation)). |
| Image: *This Ollama version cannot generate images* | Expected with Ollama ≥ 0.32.6, which temporarily removed image generation; the app falls back to the free APIs automatically. |
| Port 8080 is busy | `PORT=9000 python main.py` |

---

Dataset: arXiv metadata (CC0). Code: [MIT](LICENSE).

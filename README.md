# TrendScout AI

TrendScout AI is an evidence-first market intelligence system for AI startups. It ingests news from RSS feeds, extracts time-stamped business events (funding, hiring, product launches, regulation, etc.), builds a knowledge graph in Neo4j, and answers questions like "Why is XYZ trending?" with citations back to source URLs.

## Why this project

Most "trend" dashboards are hard to audit: they blend sentiment, hype, and volume and rarely show verifiable evidence. TrendScout is designed to be repeatable and defensible:
- "Trending" is treated as attention/volume, not "good news" by default.
- Event signals are stored as a time-stamped ledger with confidence + polarity.
- Answers are grounded in cited snippets and graph relationships.

## What it does

- Ingests AI/startup RSS sources and normalizes article records.
- Chunks articles into snippets with deterministic IDs (`article_id`, `snippet_id`).
- Runs LLM extraction to produce structured Entity/KPI/SWOT rows with evidence.
- Builds a two-layer knowledge graph:
  - Article graph for themes/clusters and conversation centrality
  - KPI/event layer for entity-linked, time-stamped signals (funding, hiring, risk)
- Computes temporal features (30/90/180-day windows) and composite rankings.
- Supports hybrid retrieval (graph + vector search) for grounded Q&A.

## System architecture (brief)

```
RSS Feeds
  -> articles_raw.parquet
    -> snippets.parquet
      -> kpi_entities.parquet
        -> Neo4j (KG) + Chroma (Vector Index)
          -> Streamlit UI
```

## Tech stack

- Python (pipeline, scoring, evaluation)
- Neo4j (knowledge graph)
- ChromaDB (vector store) + `sentence-transformers` (embeddings)
- Local LLM inference for extraction (Ollama or Hugging Face `transformers`)
- Parquet (local artifacts in `DATA/`)
- Streamlit (demo UI)

## Quick start

### Prerequisites

- Python 3.10-3.12
- Neo4j running locally or in the cloud
- Either:
  - Ollama running locally (model configurable; default is `llama3.1` in `CODE/config.py`), or
  - A Hugging Face local model available to `transformers` (for HPC/offline runs)

### Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set Neo4j credentials:
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

### Run the pipeline

```bash
# clean run (rebuilds local artifacts + Neo4j state)
python CODE/run_pipeline.py --clean

# incremental run (ingests/processes new items)
python CODE/run_pipeline.py
```

### Verify the system

```bash
python CODE/verify.py
```

## Run the app (Streamlit)

```bash
streamlit run CODE/app.py
```

## Testing

```bash
python -m unittest discover -v -s tests
```

## Repo notes

- `DATA/` contains generated artifacts and is not meant to be committed (large files, reproducible from RSS, and may contain copyrighted text).
- Cleanup helper: `powershell -ExecutionPolicy Bypass -File scripts/cleanup_local_artifacts.ps1`

## Running extraction without Ollama (Hugging Face, no API)

If you want to run extraction on an HPC GPU node without relying on Ollama, you can use the Hugging Face backend:

```bash
python CODE/extract_llm.py --provider hf --hf-model Qwen/Qwen2.5-14B-Instruct --out DATA/kpi_entities.parquet
```

For higher throughput on GPUs, you can use vLLM (recommended on HPC):

```bash
python CODE/extract_llm.py --provider hf --hf-backend vllm --hf-model Qwen/Qwen2.5-14B-Instruct --out DATA/kpi_entities.parquet
```

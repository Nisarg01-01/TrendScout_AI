# TrendScout AI

TrendScout AI is an evidence-first market intelligence system for AI startups. It ingests news from RSS feeds, extracts time-stamped business events (funding, hiring, product launches, regulation, etc.), builds a knowledge graph in Neo4j, and answers questions like “Why is XYZ trending?” with citations back to source URLs/snippets.

## What you get

- News ingestion + preprocessing into stable, de-duplicated snippets.
- LLM extraction into structured `Entity` / `KPI` / `SWOT` rows (with evidence strings).
- Neo4j knowledge graph + article clustering + entity ranking.
- Chroma vector index for RAG.
- Streamlit UI for Q&A over RAG + KG with “Unknown” when evidence is missing (no hallucinations by design).

## Repo layout (high level)

```
RSS feeds
  -> DATA/articles_raw.parquet
    -> DATA/snippets.parquet
      -> DATA/kpi_entities.parquet
        -> Neo4j (KG) + CODE/chroma_db (Vector Index)
          -> Streamlit UI
```

## Quickstart (clone → run)

### 1) Clone

```bash
git clone <your-repo-url>
cd TrendScout_AI
```

### 2) Create an environment + install deps

Using `pip`:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Or using Conda:

```bash
conda create -n trendscout python=3.11 -y
conda activate trendscout
pip install -r requirements.txt
```

### 3) Configure `.env`

Copy `.env.example` to `.env` and set:
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

Optional (local LLM via Ollama):
- Ensure Ollama is running and the model exists (defaults are in `CODE/config.py`).

### 4) Run the pipeline (fresh vs incremental)

TrendScout supports two modes:
- **Fresh**: delete/rebuild outputs from scratch.
- **Incremental**: append only new data (recommended for day-to-day runs).

Fresh end-to-end (rebuilds local parquet outputs + Chroma; also reset Neo4j when you run graph build with `--reset`):

```bash
python CODE/reset_data.py --all --yes
python CODE/ingest_news.py --fresh
python CODE/preprocess.py --fresh
python CODE/extract_llm.py --provider hf --hf-backend transformers --hf-model Qwen/Qwen2.5-14B-Instruct --out DATA/kpi_entities.parquet --fresh
python CODE/graph_build.py --reset --yes
python CODE/analysis_article_communities.py
python CODE/ranking_engine.py
python CODE/temporal_features.py
python CODE/load_chroma.py --rebuild
```

Incremental daily run (keeps existing data and appends new items):

```bash
python CODE/ingest_news.py
python CODE/preprocess.py
python CODE/extract_llm.py --provider hf --hf-backend transformers --hf-model Qwen/Qwen2.5-14B-Instruct --out DATA/kpi_entities.parquet
python CODE/graph_build.py
python CODE/analysis_article_communities.py
python CODE/ranking_engine.py
python CODE/temporal_features.py
python CODE/load_chroma.py
```

### 5) Run Streamlit

```bash
streamlit run CODE/app.py
```

## Testing

```bash
pytest -q
```

## Notes

- `DATA/` contains generated artifacts and is not meant to be committed (large files, and may contain copyrighted text).
- `CODE/chroma_db` is a local vector DB directory; delete/rebuild it only when you want a fresh index.

## HPC (optional)

You can run extraction on an HPC GPU node using Hugging Face models:

```bash
python CODE/extract_llm.py --provider hf --hf-backend transformers --hf-model Qwen/Qwen2.5-14B-Instruct --out DATA/kpi_entities.parquet
```

If you use `vllm`, set cache dirs to scratch to avoid filling your home quota:

```bash
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=/scratch/$USER/xdg_cache
mkdir -p "$HF_HUB_CACHE" "$XDG_CACHE_HOME"
```

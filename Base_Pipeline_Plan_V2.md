# Base Pipeline Plan (V2)

## 1. Folder Structure
```
trendscout/
  ingest_news.py
  ingest_jobs.py
  preprocess.py
  extract_llm.py
  dedupe_entities.py
  graph_build.py
  rag_index.py
  chat_service.py
  dashboard/
  data/
```

---

## 2. File-by-File Details

### ingest_news.py
Fetch RSS feeds, dedupe by link, append new rows.
**Output:** data/articles_raw.parquet  
Columns: [source, title, link, summary, published, fetched_at]

### ingest_jobs.py
Fetch GitHub AI startup jobs, dedupe by URL, append.
**Output:** data/jobs_raw.parquet  
Columns: [company, title, location, url, posted_at, fetched_at]

### preprocess.py
Clean and split into snippets.
**Output:** data/snippets.parquet  
Columns: [source, title, link, snippet_id, text, published]

### extract_llm.py
Use LLM to extract entities, KPIs, stance.
**Output:** data/kpi_entities.parquet  
Columns: [snippet_id, entity, entity_type, kpi_type, stance, confidence]

### dedupe_entities.py
Canonicalize names and merge similar ones (RapidFuzz ≥90).
**Output:** data/entity_map.parquet

### graph_build.py
Merge new data into Neo4j graph:
Articles → CO_LINK; Snippets → SIM edges; Entities → MENTIONS + inferred edges.
Append-safe MERGE operations.

### rag_index.py
Create Chroma vector collections (snippets, jobs).
Metadata: url, title, entity_ids, kpi, cluster_id, date.

### chat_service.py
LangChain RetrievalQA: intent classify → retrieve → generate → citations.

### dashboard/
Streamlit dashboard: clusters, startups, jobs.

---

## 3. Append Logic
- Parquet files append new data only.
- Neo4j MERGE prevents duplicates.
- Chroma dedupes by document id.

---

## 4. Data Lineage
```
RSS/Jobs
   ↓
Ingest (.parquet append)
   ↓
Preprocess (snippets)
   ↓
LLM extraction (entities + KPIs)
   ↓
Deduplication
   ↓
Neo4j (Gᵃ + Gᵏ + relations)
   ↓
Chroma (vectors)
   ↓
LangChain Chat / Dashboard
```

---

## 5. Defaults
Louvain clustering | TF-IDF ≥ 0.30 | RapidFuzz ≥ 90 | RAG top-k = 5.  
LangGraph planned for future agent workflows.

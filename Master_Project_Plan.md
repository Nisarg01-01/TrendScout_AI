# TrendScout — Master Project Plan

## 1. Project Overview & Alignment
**TrendScout** is an AI-powered market intelligence platform designed to track startup trends, funding, and innovations.

This plan aligns with:
- **Professor’s Guidance:** Article & KPI graphs, clustering, ranking, RAG, and temporal layers.
- **Group Proposal:** Structured/unstructured ETL, Neo4j + Postgres + LLM reasoning.
- **Vision:** Continuous data ingestion, knowledge graph, ranking, chat & dashboard.

**Technology Stack:**
| Layer | Tool |
| ------ | ---- |
| **Graph Store** | Neo4j AuraDB (Cloud Shared) |
| **Feature Store** | Local Parquet Files (`data/*.parquet`) |
| **Vector Store** | ChromaDB (Local Persistent) |
| **Orchestration** | Python Scripts (Microservices) |
| **UI** | Streamlit |
| **LLM** | Ollama (Llama 3.2) |

---

## 2. System Architecture & Data Lineage

The system follows a linear ETL pipeline with a "Bootstrap" option for historical data:

```
[Optional: Bootstrap Historical Data]
   ↓
RSS/Jobs (Ingestion)
   ↓
Preprocess (Clean & Chunk)
   ↓
LLM Extraction (Entities, KPIs, SWOT)
   ↓
Deduplication (Entity Resolution)
   ↓
Graph Construction (Neo4j Cloud)
   ↓
Analytics (Community Clustering, Ranking, Forecasting)
   ↓
Vector Indexing (Parquet Embeddings)
   ↓
Vector Loading (ChromaDB)
   ↓
Retrieval Service (Hybrid RAG: Vector + Graph + Analytics)
   ↓
Streamlit UI
```

---

## 3. Folder Structure

```
trendscout/
  ingest_news.py
  ingest_jobs.py
  preprocess.py
  extract_llm.py
  dedupe_entities.py
  graph_build.py
  analysis_article_communities.py
  analysis_community_swot_summary.py
  rag_index.py
  load_chroma.py
  retrieval_service.py
  app.py
  run_pipeline.py
  data/
  debug/
    bootstrap_techcrunch_2025.py
    preprocess_bootstrap.py
```

---

## 4. Implementation Pipeline (File-by-File)

### 4.1 Data Ingestion
**Goal:** Fetch data and append to raw storage.

- **`ingest_news.py`**
    - **Sources:** TechCrunch, VentureBeat, TLDR (AI section), etc.
    - **Logic:** Fetch RSS feeds, dedupe by link, filter for AI keywords, append new rows.
    - **Output:** `data/articles_raw.parquet`

- **`ingest_jobs.py`**
    - **Sources:** GitHub repo listing AI-startup jobs.
    - **Logic:** Scrape markdown table, dedupe by URL, append.
    - **Output:** `data/jobs_raw.parquet`

- **`debug/bootstrap_techcrunch_2025.py`** (Bootstrap Mode)
    - **Logic:** Scrapes historical 2025 TechCrunch articles to seed the system.
    - **Output:** `data_bootstrap/techcrunch_2025_bootstrap.parquet`

### 4.2 Preprocessing
- **`preprocess.py`**
    - **Logic:** Clean text and split into snippets.
    - **Output:** `data/snippets.parquet`

- **`debug/preprocess_bootstrap.py`** (Bootstrap Mode)
    - **Logic:** Loads bootstrap data, chunks it, and appends to snippets.

### 4.3 Intelligence Layer
- **`extract_llm.py`**
    - **Logic:** Use LLM to extract entities, KPIs, and stance (+ / − / 0).
    - **New Extraction Fields:**
        - **Industry/Domain:** (e.g., "Healthcare", "Fintech", "Generative AI") for sector filtering.
        - **SWOT Elements:** Tag snippets as Strength, Weakness, Opportunity, or Threat.
    - **Output:** `data/kpi_entities.parquet`

- **`dedupe_entities.py`**
    - **Logic:** Canonicalize names (RapidFuzz ≥ 90), merge exact matches.
    - **Output:** `data/entity_map.parquet`

### 4.4 Knowledge Graph & Analytics
- **`graph_build.py`**
    - **Logic:** Pushes nodes (Entity, Article, Industry) and relationships (MENTIONED_IN, IN_INDUSTRY) to **Neo4j AuraDB**.
    - **Output:** Neo4j Database (Cloud).

- **`analysis_article_communities.py`**
    - **Logic:** Groups articles into thematic communities using Louvain clustering.
    - **Output:** `data/article_communities.parquet`

- **`analysis_community_swot_summary.py`**
    - **Logic:** Aggregates SWOT counts, computes Trend Rankings (Score + Slope), and generates Forecasts.
    - **Output:** `data/community_*.parquet`, `data/entity_ranking.parquet`

### 4.5 Vector Search
- **`rag_index.py`**
    - **Logic:** Generates embeddings for all snippets using Ollama.
    - **Output:** `data/snippets_embeddings.parquet` (Portable file).

- **`load_chroma.py`**
    - **Logic:** Reads the embeddings parquet and loads it into a local **ChromaDB** collection.
    - **Output:** `chroma_db/` (Local Vector Database).

### 4.6 Retrieval & UI
- **`retrieval_service.py`**
    - **Logic:** Hybrid RAG engine.
        1.  **Vector**: Queries ChromaDB for semantic context.
        2.  **Graph**: Queries Neo4j for relationships.
        3.  **Analytics**: Reads Parquet for Rankings/Forecasts.
        4.  **Synthesis**: LLM generates answer using "Cross-Reference" prompt.

- **`app.py`**
    - **Logic:** Streamlit dashboard for user interaction. Renders CSV tables for rankings.

### 4.4 Graph Construction
- **`graph_build.py`**
    - **Logic:** Load `kpi_entities.parquet` and `entity_map.parquet`.
    - **Target:** Neo4j Aura (Cloud).
    - **Nodes:** `Article`, `Entity`, `Industry`.
    - **Edges:** `MENTIONS`, `BELONGS_TO`.
    - **Status:** Completed (Neo4j Aura connected).

### 4.5 Vector Indexing
- **`rag_index.py`**
    - **Logic:** Generate embeddings for snippets using Llama 3.2.
    - **Target:** Local Parquet.
    - **Output:** `data/snippets_embeddings.parquet`.
    - **Status:** Completed.

### 4.6 Application Layer
- **`retrieval_service.py`**
    - **Logic:** Hybrid RAG engine.
        1.  **`VectorStore`:** Semantic search on `snippets_embeddings.parquet`.
        2.  **`GraphStore`:** Cypher queries for competitors and industries.
        3.  **`CommunityAnalytics`:** Aggregated SWOT, Trends, and Rankings from Parquet.
    - **Capabilities:**
        - **SWOT Generation:** Aggregate "Strength/Weakness" snippets.
        - **Trend Analysis:** "Cluster X grew 40% this month."
        - **Ranking:** "Top 10 Trending Startups."

- **`app.py`**
    - **Logic:** Streamlit dashboard showing clusters, top startups, and chat interface.

### 4.7 Automation & Orchestration (New)
- **`scheduler.py`**
    - **Goal:** Run the pipeline continuously to capture temporal trends.
    - **Logic:**
        - Loop every X hours.
        - Run `ingest_*.py` (fetches only *new* data since last run).
        - Trigger `preprocess` -> `extract` -> `graph` for the delta.
        - Update `last_run` timestamp in `config.json`.

---

## 5. Key Algorithms & Logic

### 5.1 Ranking Algorithm
Score startups based on graph centrality and sentiment:
```
Score = α·centrality(Gᵃ) + β·(pos−neg) + γ·edge_quality + δ·recency + ε·community_growth
```
*   **Community Growth:** Derived from Louvain clusters (is this startup in a growing trend?).

### 5.2 Append Logic
- **Parquet:** Append new data only.
- **Neo4j:** Use `MERGE` to prevent duplicates.
- **Vector Store:** Dedupe by snippet ID.

### 5.3 Entity Deduplication Strategy
1. Canonicalize (name cleanup + title-case).
2. Exact MERGE on canonical form.
3. RapidFuzz ≥ 90 for fuzzy matching.
4. Skip merge if entity types differ.

---

## 6. Roadmap
1. **Ingestion** (Done: `ingest_news.py`, `ingest_jobs.py`)
2. **Preprocess** (Next Step)
3. **LLM Extraction** (Entities, KPIs, Stance)
4. **Dedupe & Graph Build** (Neo4j)
5. **RAG Index** (Chroma)
6. **Dashboard & Chat** (Streamlit + LangChain)
7. **Advanced:** Temporal windows + forecasting (Next Cycle)

---

## 7. Detailed System Explanation (What We Have Built)

We have built a **"Logical Microservices"** architecture where each stage of the pipeline is a standalone script that communicates via the file system (Parquet files). This allows for modular development, easy debugging, and team collaboration without requiring everyone to run heavy LLM tasks.

### 7.1 The "Logical Microservices" Architecture

| Service Layer | Script Name | Role | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Seed** | `debug/bootstrap_...py` | **Historical Loader**. Loads past data (e.g., 2024-2025) to jumpstart the graph. | Static JSON/CSV | `articles_raw.parquet` |
| **2. Ingestion** | `ingest_news.py` | **News Fetcher**. Polls RSS feeds (TechCrunch, etc.) for *new* articles. | RSS Feeds | `articles_raw.parquet` (Appends) |
| **3. Processing** | `preprocess.py` | **Cleaner**. Cleans HTML, chunks text into snippets. | `articles_raw` | `snippets.parquet` |
| **4. Extraction** | `extract_llm.py` | **The Brain**. Uses LLM to find Companies, KPIs, and SWOT analysis. | `snippets` | `kpi_entities.parquet` |
| **5. Graph** | `graph_build.py` | **The Connector**. Builds the Knowledge Graph (Articles ↔ Entities). | `kpi_entities` | Neo4j Database |
| **6. Analysis** | `analysis_...py` | **The Analyst**. Detects communities (Louvain) and aggregates SWOT trends. | Neo4j + Parquet | `community_*.parquet` |
| **7. Indexing** | `rag_index.py` | **The Librarian**. Creates vector embeddings for semantic search. | `snippets` | `snippets_embeddings.parquet` |
| **8. UI/API** | `app.py` / `retrieval_service.py` | **The Interface**. Chatbot and Dashboard for users. | All Parquet + Neo4j | User Interface |

### 7.2 Key Innovations
1.  **Data Bus Strategy**: We use the `data/` folder as a shared bus. Heavy processing (LLM extraction) happens once; the results (`.parquet`) are zipped and shared with the team. This allows teammates to run the UI and Graph layers without needing API keys or GPUs.
2.  **Community Detection**: We don't just list startups; we cluster them into "Communities" (e.g., "Generative AI Healthcare") using graph algorithms, allowing us to track trends at a sector level.
3.  **Hybrid Retrieval (RAG)**: Our chat system uses **Vector Search** (for specific text matches) AND **Graph Search** (for finding competitors and relationships), providing richer answers than standard RAG.

---

## 8. Gap Analysis (Professor's Feedback)

Status check against the specific suggestions provided by the professor:

| Suggestion | Status | Notes |
| :--- | :--- | :--- |
| **Article & KPI Graphs** | ✅ **Done** | `graph_build.py` successfully models Articles, Entities, and their relationships in Neo4j. |
| **Clustering** | ✅ **Done** | `analysis_article_communities.py` implements Louvain clustering to group related articles/entities. |
| **RAG (Retrieval Augmented Gen)** | ✅ **Done** | `retrieval_service.py` implements a hybrid RAG system (Vector + Graph). |
| **Ranking** | ✅ **Done** | Implemented weighted scoring formula (`Score = α·mentions + β·stance + γ·growth`) in `analysis_community_swot_summary.py`. |
| **Temporal Layers** | ✅ **Done** | Implemented linear regression forecasting for community growth trends in `analysis_community_swot_summary.py`. |

### Missing Items to Address:
*   **None.** All core requirements and professor suggestions have been addressed.

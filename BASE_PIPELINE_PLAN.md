# TrendScout Base Pipeline Plan

## 1. Objective
Implement the base version of the pipeline that ingests TechCrunch RSS data, processes it into a knowledge graph, computes startup rankings, and exposes a simple API.

---

## 2. Folder Structure
```
TrendScout_AI/
│
├── ingest_tc.py           # RSS ingestion
├── preprocess.py          # HTML clean, snippet split
├── ner_kpi.py             # Entity + KPI extraction
├── graph_build.py         # Build article/snippet graphs
├── score_rank.py          # Scoring + ranking
├── api_app.py             # FastAPI for queries
├── check_connections.py   # Verify DB connections
├── requirements.txt
├── .env
├── SETUP_GUIDE.md
└── BASE_PIPELINE_PLAN.md
```

---

## 3. Pipeline Modules

### 1. ingest_tc.py
**Purpose:** Collect TechCrunch startup articles through legal RSS.  
**Steps:**
- Parse `https://techcrunch.com/startups/feed/` using `feedparser`.
- Extract `title`, `url`, `date`, `author`, HTML body.
- Clean tags with BeautifulSoup.
- Save `data/articles_raw.parquet`.
- Upload to shared Google Drive.

### 2. preprocess.py
**Purpose:** Normalize and prepare text.  
**Steps:**
- Load raw articles.
- Lowercase, remove HTML noise.
- Split text into 100–150 word snippets.
- Save `data/snippets.parquet`.

### 3. ner_kpi.py
**Purpose:** Extract entities and KPIs.  
**Steps:**
- Use spaCy NER (ORG, MONEY, DATE, PERSON).
- Detect KPIs (Funding, Hiring, Product, Risk) via rules or zero-shot classification.
- Assign stance (+ / − / 0).
- Save entities and KPI labels.

### 4. graph_build.py
**Purpose:** Construct knowledge graphs.  
**Steps:**
- Article Graph (Gᵃ): co-mention edges weighted by shared entities + recency decay.
- Snippet Graph (Gᵏ): similarity edges using embeddings.
- Apply Louvain clustering.
- Write nodes/edges to Neo4j.

### 5. score_rank.py
**Purpose:** Rank startups inside clusters.  
**Steps:**
- Compute PageRank/centrality per cluster.
- Sum weighted KPI stance with recency decay.
- Formula:
  ```
  Score = 0.35*Centrality + 0.4*KPI_Stance + 0.15*Recency + 0.1*InvestorQuality
  ```
- Write top scores to Supabase `rankings` table.

### 6. api_app.py
**Purpose:** Expose FastAPI endpoints.  
**Endpoints:**
- `/clusters` – list all clusters.
- `/rankings` – ranked startups per cluster.
- `/startup/{id}` – KPI scorecard.
- `/chat` – RAG-style Q&A (later integration).

---

## 4. Data Flow
```
RSS → ingest_tc.py → preprocess.py → ner_kpi.py → graph_build.py → score_rank.py → api_app.py
```

---

## 5. Outputs
- `articles_raw.parquet`
- `snippets.parquet`
- `entities.parquet`
- `kpi_labels.parquet`
- Neo4j graph database
- PostgreSQL ranking table
- Working FastAPI service

---

## 6. Future Expansion Hooks
- Extend ingestion to Crunchbase/VentureBeat.
- Add ML-based KPI detection.
- Integrate time-based metrics and forecasting.
- Add LLM reasoning in chat endpoint.
- Build Streamlit analytics dashboard.

---
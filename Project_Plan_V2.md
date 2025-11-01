# TrendScout — Project Plan (V2)

## 1. Alignment Summary
This plan is fully aligned with:
- **Professor’s plan** – article and KPI graph, clustering, ranking, RAG, temporal layer.  
- **Group proposal** – structured/unstructured ETL, Neo4j + Postgres + LLM reasoning, conversational interface.  
- **TrendScout vision** – continuous data ingestion, knowledge graph, ranking, chat & dashboard.

LangChain + Chroma are active; LangGraph is scheduled for the next cycle.

---

## 2. Data Sources
1. **News:** TechCrunch RSS, VentureBeat RSS, TLDR (AI section).  
2. **Jobs:** GitHub repo listing AI-startup jobs (only this source).  
3. **KPI / Stance:** LLM detects KPI type + stance (+ / − / 0) with keyword backstop.

Each ingestion appends new entries when re-run, ensuring incremental growth.

---

## 3. Graph Model

### 3.1 Article Graph (Gᵃ)
**Purpose:** Cluster articles by shared startup mentions.

**Nodes**
`:Article {link, title, published, source, centrality, cluster_id}`  
**Edges**
`(:Article)-[:CO_LINK {w}]->(:Article)` *w* = shared startups.  
**Algorithms**
Louvain → `cluster_id` | PageRank → `centrality`.

---

### 3.2 KPI / Snippet Graph (Gᵏ)
**Purpose:** Group evidence snippets per KPI type inside each article cluster.

**Nodes**
`:Snippet {snippet_id, text, link}` | `:KPI {type}` ∈ {Funding, Product, Partnership, Hiring, Risk, Acquisition}  
**Edges**
```
(:Snippet)-[:IN]->(:Article)
(:Snippet)-[:ABOUT {stance, confidence}]->(:KPI)
(:Snippet)-[:SIM {score}]->(:Snippet) # TF-IDF cos ≥ 0.30, same KPI
```
**Clustering:** Louvain → sub-themes with polarity balance + recency.

---

### 3.3 Entity & Relations Layer
**Nodes**
`:Entity {name, kind, is_ai, canonical_name}`  
**Relations**
- `:Article-[:MENTIONS]->:Entity`
- LLM-inferred edges  
  `INVESTED_IN`, `PARTNERED_WITH`, `ACQUIRED`, `COMPETES_WITH` {reason, confidence, snippet_id, published}

---

## 4. Entity Deduplication
1. Canonicalize (name cleanup + title-case).  
2. Exact MERGE on canonical form.  
3. RapidFuzz ≥ 90 if needed.  
4. Optional `aliases` property or `:ALIAS_OF`.  
5. Skip merge if type differs.  
Appended data merges automatically via Neo4j `MERGE` operations.

---

## 5. Retrieval and Chat (RAG)
**Retriever:** BM25 + TF-IDF rerank → top-k snippets.  
**Generator:** LLM (any backend).  
**Citations:** Always URL + title.  
**Decision:** Keyword → LLM fallback (news / jobs / both).  
**Indexes:** News (Gᵏ) and Jobs (GitHub) collections in Chroma.

---

## 6. Ranking
```
Score = α·centrality(Gᵃ) + β·(pos−neg) + γ·edge_quality + δ·recency
```
AI-startups only; recency uses time decay → temporal windows later.

---

## 7. Visualization & Interaction
### 7.1 Dashboard (Streamlit)
Clusters → Top startups, KPI counts.  
Startup → KPI trend + supporting snippets.  
Jobs → AI startup roles (list only).

### 7.2 Chat
LangChain RAG pipeline + Chroma retriever + LLM answers with citations.  
LangGraph (next cycle) → structured decision flows.

---

## 8. Technology Overview
| Layer | Tool |
| ------ | ---- |
| Graph Store | Neo4j Aura |
| Feature Store | Supabase PostgreSQL |
| Vector Store | Chroma DB |
| Orchestration | LangChain |
| Future Agent Logic | LangGraph |
| UI | Streamlit |
| LLM | Generic (local or API) |

---

## 9. Roadmap
1. Ingestion (TechCrunch, VentureBeat, TLDR, GitHub Jobs)  
2. Preprocess (clean + snippets)  
3. LLM Extraction (entities, KPIs, stance)  
4. Dedupe → KG build (Gᵃ + Gᵏ + relations)  
5. RAG index (Chroma) + Chat API  
6. Dashboard (Streamlit)  
7. Temporal windows + forecasting (next cycle)

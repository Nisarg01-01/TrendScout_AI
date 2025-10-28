# Project_Plan

## Part 1: Base Pipeline (Your MVP)

### Purpose
Deliver a working end-to-end system that ingests TechCrunch RSS data legally, processes it, builds graphs, ranks startups, and exposes a minimal API and interface. This becomes the foundation for the full system.

### Pipeline Blocks

| Stage | Task | Key Output |
|--------|------|------------|
| 1. Data Ingestion (RSS-safe) | Use feedparser on TechCrunch feeds; pull title, link, date, body | articles_raw.parquet |
| 2. Preprocessing | Clean HTML, split paragraphs/snippets, detect language, normalize | snippets.parquet |
| 3. Entity & KPI Extraction | spaCy + custom NER for startups, investors, sectors; stance classification | entities.parquet, kpi_labels.parquet |
| 4. Article Graph (Gᵃ) | Nodes=articles; edges=co-mentions weighted by shared entities × recency decay; Louvain clustering | Neo4j Article/Cluster/CO_LINK |
| 5. Snippet/KPI Graph (Gᵏ) | Embed snippets, connect by similarity and KPI type; HDBSCAN clustering | Neo4j Snippet/KPI/KPICluster |
| 6. Scoring | Per-startup composite: Score = α·Centrality + β·KPI_Stance + γ·Recency | rankings.parquet |
| 7. Storage + API | Store graphs in Neo4j, features in Postgres, snippets in Chroma; serve FastAPI endpoints | Running service |
| 8. UI & Eval | Streamlit dashboard for clusters and top startups; evaluate modularity, silhouette, faithfulness | Demo app + metrics notebook |

### Base-Pipeline Outputs
- Neo4j graph with article + snippet clusters  
- Chroma vector store of snippets  
- PostgreSQL tables: startup_features_daily, rankings  
- Artifacts: articles_raw.parquet, entities.parquet, snippets.parquet  
- API + minimal dashboard  
- Demonstrable query-to-answer capability with citations

---

## Part 2: Full-Scale Project (Team Expansion Plan)

### Goal
Extend the MVP into a research-grade Trend-Knowledge Graph + LLM forecasting system.

### Team Structure (2-person pods)
| Team | Focus | Depends On |
|-------|--------|------------|
| A. Data & Ingestion | Add sources (Crunchbase API, VentureBeat, etc.) | Base pipeline |
| B. NLP & Ontology | Expand NER, KPI taxonomy, stance accuracy | A |
| C. Graph & Ranking | Optimize algorithms, investor weighting | A,B |
| D. Temporal & Forecasting | Build time-series, survival/ranking models, SHAP | C |
| E. RAG & Chat Layer | Build LangGraph chain (Cluster → KPI → Forecast → Explain) | C,D |
| F. Evaluation & UI | Create dashboards, human eval, visualization | C–E |

### Expansion Roadmap
| Phase | Enhancement | Output |
|--------|--------------|--------|
| 1 | Multi-source ingestion | Broader article graph |
| 2 | Ontology & stance refinement | Higher stance accuracy |
| 3 | Temporal graph engine | Time-aware KG |
| 4 | Forecast layer | Predictive momentum scores |
| 5 | LLM reasoning layer | Chat with citations |
| 6 | Visualization & insight UI | Analyst dashboard |
| 7 | Evaluation & calibration | Measured quality |

### End-State Deliverables
- Unified Knowledge Graph + Feature Store (startups, investors, KPIs, time)  
- Momentum ranking and forecasting engine with SHAP explanations  
- RAG-powered chat assistant for analysts  
- Evaluation framework and reproducible reports

### Summary
| Stage | Owner | Output |
|--------|--------|--------|
| You (Base) | RSS→KG pipeline + API/UI | Working prototype |
| Team (Expansion) | Enriched data, ontology, temporal + forecast + LLM chat | Complete TrendScout system |

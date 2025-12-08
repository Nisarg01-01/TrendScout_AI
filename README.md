# TrendScout AI

Market intelligence system that analyzes startup trends, funding patterns, and industry signals using a two-layer knowledge graph architecture.

## Overview

Two-Layer Graph Architecture: Article Graph (Gᵃ) + KPI Graph (Gᵏ)
- Structured KPI Extraction: Funding, hiring, partnerships, products
- 4-Component Ranking System: Centrality + KPI Stance + Recency + Investor Quality
- Temporal Analysis: 30/90/180-day rolling windows
- RAG-Powered Q&A: Vector search + graph traversal + LLM synthesis

## Architecture

### Data Layer
- Neo4j: Knowledge graph (entities, articles, clusters, investors)
- ChromaDB: Vector embeddings for semantic search
- Parquet Files: Processed data and temporal features

### Graph Schema
```
Article Graph (Gᵃ):
├── Article nodes
├── Entity nodes
├── Cluster nodes (Louvain communities)
├── CO_LINK edges (Jaccard similarity × recency decay)
└── RANKED_IN edges (entities ranked within clusters)

KPI Graph (Gᵏ):
├── Snippet nodes
├── KPI nodes (funding, hiring, etc.)
├── KPICluster nodes (HDBSCAN within article clusters)
├── Investor nodes (with prestige scores)
├── SIMILAR_TO edges (snippet similarity)
└── FUNDED_BY edges (entity → investor)
```

### Ranking Formula
```
Score = 0.3×Centrality + 0.4×KPI_Stance + 0.2×Recency + 0.1×Investor_Quality
```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt

# Neo4j: localhost:7687
# Ollama: localhost:11434 with llama3.1 model
```

### Run Pipeline
```bash
# Clean run (recommended for first time)
python run_pipeline.py --clean

# Normal run (incremental)
python run_pipeline.py
```

Duration: 30-60 minutes

### Verify Installation
```bash
python verify.py
```

### Launch Web Interface
```bash
streamlit run app.py
```

Access at: http://localhost:8501

## Pipeline Steps

1. **Data Ingestion**
   - `ingest_news.py` - Fetch TechCrunch articles
   - `ingest_jobs.py` - Extract job postings

2. **Processing**
   - `preprocess.py` - Clean and chunk text into snippets
   - `extract_llm.py` - Extract structured KPIs with LLM (optimized: 4 workers)
   - `dedupe_entities.py` - Merge duplicate entity mentions

3. **Graph Construction**
   - `graph_build.py` - Build Article Graph with CO_LINK edges
   - `analysis_article_communities.py` - Louvain clustering → Clusters

4. **KPI & Enrichment**
   - `kpi_clustering.py` - Build KPI Graph (HDBSCAN on snippets)
   - `investor_extraction.py` - Extract investors, assign prestige scores
   - `temporal_features.py` - Calculate rolling window features

5. **Analysis & Ranking**
   - `analysis_community_swot_summary.py` - SWOT aggregation
   - `ranking_engine.py` - PageRank + composite scoring

6. **Indexing**
   - `rag_index.py` - Prepare graph for retrieval
   - `load_chroma.py` - Index snippets in ChromaDB

## Testing

### Graph Schema
```cypher
// In Neo4j Browser (localhost:7474)

// Check node counts
MATCH (a:Article) RETURN count(a) as Articles
MATCH (e:Entity) RETURN count(e) as Entities
MATCH (c:Cluster) RETURN count(c) as Clusters
MATCH (kc:KPICluster) RETURN count(kc) as KPIClusters
MATCH (i:Investor) RETURN count(i) as Investors

// View top-ranked entities
MATCH (e:Entity)-[r:RANKED_IN]->(c:Cluster)
RETURN e.name, r.rank, r.score, r.centrality, r.kpi_stance, 
       r.recency, r.investor_quality
ORDER BY r.rank LIMIT 10

// Explore investor connections
MATCH (e:Entity)-[:FUNDED_BY]->(i:Investor)
RETURN e.name, i.name, i.prestige
ORDER BY i.prestige DESC LIMIT 20
```

### Data Files
```bash
# Check generated files in DATA directory
ls DATA/

# Expected files:
# - articles.parquet
# - snippets.parquet
# - kpi_entities.parquet
# - entity_map.parquet
# - temporal_features.parquet
```

## Project Structure

```
TrendScout_AI/
├── README.md
├── requirements.txt
├── CODE/
│   ├── app.py
│   ├── config.py
│   ├── run_pipeline.py
│   ├── verify.py
│   ├── ingest_news.py
│   ├── ingest_jobs.py
│   ├── preprocess.py
│   ├── extract_llm.py
│   ├── dedupe_entities.py
│   ├── graph_build.py
│   ├── analysis_article_communities.py
│   ├── kpi_clustering.py
│   ├── investor_extraction.py
│   ├── temporal_features.py
│   ├── analysis_community_swot_summary.py
│   ├── ranking_engine.py
│   ├── rag_index.py
│   ├── load_chroma.py
│   ├── retrieval_service.py
│   ├── utils/
│   │   └── neo4j_utils.py
│   └── debug/
│       ├── bootstrap_techcrunch_2025.py
│       └── preprocess_bootstrap.py
├── DATA/
└── EVALUATIONS/
```

## Configuration

Edit `config.py` to customize:

```bash
# Create .env file in project root

# For Neo4j AuraDB (Cloud)
NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="YourAuraDBPassword"

# For Local Neo4j
# NEO4J_URI="bolt://localhost:7687"
# NEO4J_USERNAME="neo4j"
# NEO4J_PASSWORD="your_local_password"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1"

# Data paths
DATA_DIR = "DATA"
```

## Use Cases

### Market Intelligence
```
"Who are the top-ranked AI startups?"
→ Shows ranked entities with scores and components
```

### Funding Analysis
```
"What is OpenAI's funding situation?"
→ Shows KPI breakdown, investors, and prestige scores
```

### Trend Detection
```
"Show me recent developments in autonomous driving"
→ Uses temporal features and recency weighting
```

### Competitive Analysis
```
"Compare Anthropic and OpenAI"
→ Multi-entity comparison with SWOT and rankings
```

## Troubleshooting

### Pipeline Fails

**At kpi_clustering.py:**
- Verify Cluster nodes exist: `MATCH (c:Cluster) RETURN count(c)`
- Check sentence-transformers: `pip install sentence-transformers`

**At investor_extraction.py:**
- May be normal if no funding KPIs extracted
- Check kpi_entities.parquet for funding data

**At temporal_features.py:**
- Normal for large datasets (progress logged)
- Can reduce date range if needed

### Neo4j Connection
- Verify Neo4j running: http://localhost:7474
- Check credentials in config.py
- Ensure port 7687 not blocked

### Streamlit Shows Old Data
- Refresh: Ctrl+Shift+R
- Restart: `streamlit run app.py`
- Check file timestamps: `ls -l DATA/*.parquet`

## Performance

### LLM Extraction
- **Speed**: ~40% faster (4 workers, optimized)
- **Rate**: 50-100 snippets/min
- **Quality**: 80-90% structured field accuracy

### Graph Operations
- **Articles**: ~1000/sec insertion
- **Clustering**: ~30 sec for 10K articles
- **KPI Clustering**: ~60 sec for 5K snippets
- **Ranking**: ~15 sec for 500 entities

### Query Response
- **Vector search**: 100-200ms
- **Graph traversal**: 50-100ms
- **LLM generation**: 2-5 sec
- Total: 3-6 sec per query

## Key Features

### Structured KPI Extraction
- **Funding**: Amount, stage, investors, date
- **Hiring**: Count, roles, skills, departments
- **Partnerships**: Partners, type, details
- **Products**: Name, features, launch date

### Temporal Analysis
- **30-day window**: Recent activity
- **90-day window**: Short-term trends
- **180-day window**: Medium-term momentum

### Investor Quality Scoring
- Prestige scores for 30+ top VCs
- Sequoia Capital: 1.0
- Andreessen Horowitz: 0.9
- Y Combinator: 0.7
- Tier 2 VCs: 0.5-0.6

### Cluster-Aware Retrieval
- Scoped vector search within clusters
- Hierarchical KPI organization
- Community-based ranking

## Documentation

See documentation files in repository for additional details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Run verification: `python verify.py`
4. Submit pull request

## License

MIT License

## Built With

- Neo4j (graph database)
- ChromaDB (vector store)
- Ollama (LLM inference)
- Streamlit (web interface)
- Sentence Transformers (embeddings)
- HDBSCAN (clustering)

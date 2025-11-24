# TrendScout AI: Microservices Architecture & Setup Guide

This document outlines the logical microservices architecture of TrendScout AI and provides step-by-step instructions for recreating the environment and sharing data with the team.

---

## 1. Logical Microservices Architecture

Although the project resides in a single directory for simplicity, each Python script acts as an independent "microservice" responsible for a specific stage of the data pipeline.

| Service Name | Script | Responsibility | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion Service** | `ingest_news.py`<br>`ingest_jobs.py` | Fetches raw data from RSS feeds and GitHub. | Internet (RSS, HTTP) | `data/articles.parquet`<br>`data/jobs.parquet` |
| **Bootstrap Service** | `debug/bootstrap_techcrunch_2025.py`<br>`debug/preprocess_bootstrap.py` | **(One-Time)** Ingests historical 2025 data to seed the system. | TechCrunch Archives | `data/articles.parquet` (Appended) |
| **Preprocessing Service** | `preprocess.py` | Cleans, deduplicates, and formats raw data. | Raw Parquet Files | `data/snippets.parquet` |
| **Extraction Service** | `extract_llm.py` | Uses LLM (Ollama) to extract Entities, KPIs, and SWOT. | `data/snippets.parquet` | `data/kpi_entities.parquet` |
| **Entity Resolution Service** | `dedupe_entities.py` | Normalizes entity names (e.g., "OpenAI Inc" -> "OpenAI"). | `data/kpi_entities.parquet` | `data/entity_map.parquet` |
| **Graph Service** | `graph_build.py` | Constructs the Knowledge Graph in Neo4j. | Extracted Data + Map | Neo4j Database |
| **Community Analysis Service** | `analysis_article_communities.py` | Clusters articles into thematic communities (Louvain). | `data/kpi_entities.parquet` | `data/article_communities.parquet` |
| **Analytics Aggregation Service** | `analysis_community_swot_summary.py` | Aggregates SWOT, trends, forecasts, and rankings. | Communities + KPI Data | `data/community_*.parquet`<br>`data/entity_ranking.parquet` |
| **Vector Service** | `rag_index.py`<br>`load_chroma.py` | Generates embeddings and loads them into ChromaDB. | `data/snippets.parquet` | `data/snippets_embeddings.parquet`<br>`chroma_db/` (Local DB) |
| **Backend API** | `retrieval_service.py` | Orchestrates RAG (Vector + Graph + Analytics) queries. | User Query | JSON/Text Response |
| **Frontend Service** | `app.py` | Streamlit UI for user interaction. | User Input | Interactive UI |

---

## 2. Step-by-Step Recreation Guide (For You)

Follow these steps to recreate the entire project environment from scratch.

### Prerequisites
1.  **Python 3.10+** installed.
2.  **Neo4j Desktop** (or AuraDB) installed and running.
3.  **Ollama** installed and running (`ollama serve`).
4.  **Llama 3.2 Model** pulled (`ollama pull llama3.2`).

### Setup Steps

1.  **Clone/Create Directory**
    ```powershell
    mkdir TrendScout_AI
    cd TrendScout_AI
    # (Copy all python scripts into this folder)
    ```

2.  **Create Virtual Environment**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    *   Open `config.py`.
    *   Update `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` to match your local Neo4j instance.

5.  **Run the Full Pipeline (with Bootstrap)**
    This command runs all "microservices" in the correct order, including fetching historical data.
    ```powershell
    python run_pipeline.py --clean --bootstrap
    ```
    *   *Note: The `--clean` flag wipes existing data. The `--bootstrap` flag injects historical data.*

6.  **Launch the App**
    ```powershell
    streamlit run app.py
    ```

---

## 3. Team Mate's Guide (How to Run)

Share this section with your team.

### Quick Start
1.  **Get the Code**: Clone the repository or unzip the project folder.
2.  **Install Requirements**:
    ```powershell
    pip install -r requirements.txt
    ```
3.  **Setup Neo4j**:
    *   Install Neo4j Desktop.
    *   Create a new project/database.
    *   Set the password to match `config.py` (default: `password`) or update `config.py`.
4.  **Setup Ollama**:
    *   Install Ollama.
    *   Run `ollama pull llama3.2`.

### ⚠️ Crucial: Sharing Data (Skip the Wait)
Running `extract_llm.py` and `rag_index.py` can take hours because they use the LLM heavily. **Do not make every team member run this from scratch.**

**How to Share Data:**
1.  **The "Master" (You)** runs the full pipeline once:
    ```powershell
    python run_pipeline.py
    ```
2.  **Zip the Data**:
    *   Locate the `data/` folder in your project.
    *   Zip the entire `data/` folder.
3.  **Share**: Send `data.zip` to your teammates (via Drive, Slack, etc.).

**How Team Mates Use Shared Data:**
1.  **Unzip**: Extract `data.zip` into their project root. They should see a `data/` folder containing `.parquet` files.
2.  **Load Vector Database**: Run the following script to populate the local ChromaDB with shared embeddings:
    ```powershell
    python load_chroma.py
    ```
3.  **Build Graph Only**: They only need to populate their local Neo4j. They should **NOT** run `run_pipeline.py`. Instead, run:
    ```powershell
    python graph_build.py
    ```
    *   *This reads the shared parquet files and pushes them to their local Neo4j.*
4.  **Run App**:
    ```powershell
    streamlit run app.py
    ```

**Summary for Team Mates:**
*   **Download Code** + **Download `data/` folder**.
*   **Install Python Libs** + **Neo4j** + **Ollama**.
*   Run `python load_chroma.py` (Fast).
*   Run `python graph_build.py` (Fast).
*   Run `streamlit run app.py`.
*   *Enjoy!*

import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles_raw.parquet")
JOBS_FILE = os.path.join(DATA_DIR, "jobs_raw.parquet")
SNIPPETS_FILE = os.path.join(DATA_DIR, "snippets.parquet")
ENTITY_MAP_FILE = os.path.join(DATA_DIR, "entity_map.parquet")
KPI_ENTITIES_FILE = os.path.join(DATA_DIR, "kpi_entities.parquet")

# Feeds
FEEDS = [
    "https://techcrunch.com/category/startups/feed/",
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://news.crunchbase.com/feed/",
    "https://tldr.tech/ai/feed",
]

# Keywords for Filtering
AI_KEYWORDS = [
    "artificial intelligence", "ai", "ai-powered", "ai-native",
    "machine learning", "ml", "deep learning", "neural network",
    "generative ai", "genai", "foundation model", "frontier model",
    "large language model", "llm", "gpt", "copilot", "agentic",
    "computer vision", "nlp", "multimodal", "autonomous",
    "ai startup", "ai company", "ai tool", "ai platform",
    "ai agent", "ai assistant", "autonomous agent", "agent swarm"
]

# Job Sources
JOBS_URL = "https://raw.githubusercontent.com/vanshb03/New-Grad-2026/main/README.md"

# Scheduler
INGEST_INTERVAL_HOURS = 6

# Preprocessing
CHUNK_SIZE = 500  # Characters
CHUNK_OVERLAP = 50

# LLM Configuration
LLM_PROVIDER = "ollama"
LLM_MODEL = "llama3.1"

# Entity Deduplication
FUZZY_THRESHOLD = 90

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Postgres
DATABASE_URL = os.getenv("DATABASE_URL")

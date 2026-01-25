import os
from typing import Dict, List
try:
    from dotenv import load_dotenv  # type: ignore
    from dotenv import dotenv_values  # type: ignore
except ModuleNotFoundError:
    load_dotenv = None
    dotenv_values = None

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(BASE_DIR, "CODE")
DATA_DIR = os.path.join(BASE_DIR, "DATA")
CHROMA_DB_PATH = os.path.join(CODE_DIR, "chroma_db")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles_raw.parquet")
JOBS_FILE = os.path.join(DATA_DIR, "jobs_raw.parquet")
SNIPPETS_FILE = os.path.join(DATA_DIR, "snippets.parquet")
ENTITY_MAP_FILE = os.path.join(DATA_DIR, "entity_map.parquet")
KPI_ENTITIES_FILE = os.path.join(DATA_DIR, "kpi_entities.parquet")


def _strip_env_value(raw: str) -> str:
    s = str(raw).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _parse_env_file(env_path: str) -> Dict[str, str]:
    """
    Minimal .env parser used as a fallback when python-dotenv isn't installed.
    Supports KEY=VALUE with optional quotes. Ignores comments/blank lines.
    """
    out: Dict[str, str] = {}
    try:
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                if s.startswith("export "):
                    s = s[len("export ") :].lstrip()
                key, val = s.split("=", 1)
                key = key.strip()
                val = _strip_env_value(val)
                if key:
                    out[key] = val
    except Exception:
        return {}
    return out


def _load_env_file():
    """
    Load .env from the repo root reliably even when scripts are run from CODE/.

    We also handle a common gotcha: if an env var is present but blank, python-dotenv
    won't override it unless override=True. Here we selectively fill only missing/blank
    values without clobbering non-empty environment configuration.
    """

    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return

    values: Dict[str, str] = {}

    if load_dotenv is not None:
        try:
            # Best-effort: still load with default semantics.
            load_dotenv(dotenv_path=env_path)
        except Exception:
            pass

    # Prefer python-dotenv's parsing when available; fallback to our minimal parser.
    if dotenv_values is not None:
        try:
            parsed = dotenv_values(env_path)  # type: ignore[call-arg]
            values = {k: v for k, v in parsed.items() if v is not None}  # type: ignore[union-attr]
        except Exception:
            values = _parse_env_file(env_path)
    else:
        values = _parse_env_file(env_path)

    for key, value in values.items():
        current = os.getenv(key)
        if current is None or str(current).strip() == "":
            os.environ[key] = _strip_env_value(value)


_load_env_file()

# Feeds
FEEDS: List[str] = [
    "https://techcrunch.com/category/startups/feed/",
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://news.crunchbase.com/feed/",
    "https://tldr.tech/ai/feed",
]

# Optional overrides/extensions for feeds (comma-separated URLs).
# - FEEDS_ONLY: if set, replaces the default FEEDS list entirely.
# - FEEDS_EXTRA: if set, appends to the default FEEDS list.
_feeds_only = os.getenv("FEEDS_ONLY")
_feeds_extra = os.getenv("FEEDS_EXTRA")
if _feeds_only and _feeds_only.strip():
    FEEDS = [u.strip() for u in _feeds_only.split(",") if u.strip()]
elif _feeds_extra and _feeds_extra.strip():
    FEEDS = FEEDS + [u.strip() for u in _feeds_extra.split(",") if u.strip()]

# Ingestion knobs (quality + scale)
MAX_ENTRIES_PER_FEED = int(os.getenv("MAX_ENTRIES_PER_FEED", "50"))
FETCH_FULL_TEXT = os.getenv("FETCH_FULL_TEXT", "1").strip() not in {"0", "false", "False", "no", "NO"}
MIN_ARTICLE_TEXT_CHARS = int(os.getenv("MIN_ARTICLE_TEXT_CHARS", "400"))

# Keywords for Filtering
AI_KEYWORDS: List[str] = [
    "artificial intelligence", "ai", "ai-powered", "ai-native",
    "machine learning", "ml", "deep learning", "neural network",
    "generative ai", "genai", "foundation model", "frontier model",
    "large language model", "llm", "gpt", "copilot", "agentic",
    "computer vision", "nlp", "multimodal", "autonomous",
    "ai startup", "ai company", "ai tool", "ai platform",
    "ai agent", "ai assistant", "autonomous agent", "agent swarm"
]

# Business/Startup context: use these to avoid pulling in consumer "AI" noise.
STARTUP_SIGNAL_KEYWORDS: List[str] = [
    "startup",
    "venture",
    "vc",
    "seed",
    "series a",
    "series b",
    "series c",
    "funding",
    "raises",
    "raised",
    "round",
    "valuation",
    "acquired",
    "acquisition",
    "ipo",
    "launch",
    "launched",
    "released",
    "announced",
    "partnership",
    "partnered",
    "customers",
    "revenue",
    "arr",
    "hiring",
    "hire",
    "layoff",
    "laid off",
]

# Common consumer/low-signal exclusions (keeps dataset "AI startups / market" focused).
EXCLUDE_KEYWORDS: List[str] = [
    "review",
    "hands-on",
    "deal",
    "deals",
    "coupon",
    "gift guide",
    "how to",
    "recipe",
    "cooking",
    "horoscope",
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

# Hugging Face (local) LLM configuration (optional, used when LLM_PROVIDER="hf" or extract_llm.py --provider hf)
# Defaults favor broad availability; override via env/CLI on HPC (e.g., Qwen2.5-14B/32B) for higher accuracy.
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_BACKEND = os.getenv("HF_BACKEND", "transformers")  # transformers | vllm (vllm not implemented here yet)
HF_TRUST_REMOTE_CODE = os.getenv("HF_TRUST_REMOTE_CODE", "0").strip() in {"1", "true", "True", "yes", "YES"}
HF_DEVICE = os.getenv("HF_DEVICE", "auto")  # auto | cpu | cuda
HF_MAX_INPUT_TOKENS = int(os.getenv("HF_MAX_INPUT_TOKENS", "3072"))
HF_BATCH_SIZE = int(os.getenv("HF_BATCH_SIZE", "1"))

# Entity Deduplication
FUZZY_THRESHOLD = 90

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Postgres
DATABASE_URL = os.getenv("DATABASE_URL")

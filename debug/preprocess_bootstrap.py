import os
import sys
import uuid
from datetime import datetime

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from preprocess import clean_text, chunk_text


BOOTSTRAP_FILE = os.path.join(ROOT_DIR, "data_bootstrap", "techcrunch_2025_bootstrap.parquet")


def normalize_published(raw: str) -> str:
    """Best-effort normalization of TechCrunch date_hint → ISO date string.

    The bootstrap file stores a loose "date_hint" string; here we try to
    parse it so temporal analysis downstream is on a clean timeline.
    On failure we simply return the original string.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""

    txt = raw.strip()

    # Common pattern in archive: "- Jan 13, 2025" or "Jan 13, 2025"
    for fmt in ["- %b %d, %Y", "%b %d, %Y", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(txt, fmt)
            return dt.date().isoformat()
        except Exception:
            continue

    # Fallback: return as-is so we at least retain ordering information
    return txt


def load_bootstrap_articles() -> pd.DataFrame:
    if not os.path.exists(BOOTSTRAP_FILE):
        raise FileNotFoundError(f"Bootstrap file not found: {BOOTSTRAP_FILE}")

    df = pd.read_parquet(BOOTSTRAP_FILE)
    if "text" not in df.columns:
        raise ValueError("Bootstrap parquet missing 'text' column")

    df = df.copy()
    df["published_norm"] = df["published"].apply(normalize_published)
    return df


def snippets_from_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bootstrap articles to snippet rows compatible with preprocess.py.

    We keep the same schema as config.SNIPPETS_FILE so downstream
    extract_llm / graph_build / rag_index work unchanged.
    """
    records = []

    for _, row in df.iterrows():
        text = clean_text(row.get("text", ""))
        if not text:
            continue

        chunks = chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    "snippet_id": str(uuid.uuid4()),
                    "source": row.get("source", "TechCrunch"),
                    "title": row.get("title", ""),
                    "link": row.get("url", ""),
                    "text": chunk,
                    "published": row.get("published_norm") or row.get("published", ""),
                    "chunk_index": idx,
                    "parent_fetched_at": None,
                }
            )

    return pd.DataFrame.from_records(records)


def append_snippets(new_snippets: pd.DataFrame):
    if new_snippets.empty:
        print("No snippets generated from bootstrap.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(config.SNIPPETS_FILE):
        existing = pd.read_parquet(config.SNIPPETS_FILE)
        combined = pd.concat([existing, new_snippets], ignore_index=True)
        combined = combined.drop_duplicates(subset=["snippet_id"])
        print(f"Existing snippets: {len(existing)}, new: {len(new_snippets)}, final: {len(combined)}")
    else:
        combined = new_snippets
        print(f"Created new snippets store with {len(combined)} rows from bootstrap.")

    combined.to_parquet(config.SNIPPETS_FILE, index=False)
    print(f"Saved snippets to {config.SNIPPETS_FILE}")


def main():
    print(f"Loading bootstrap articles from {BOOTSTRAP_FILE}...")
    df_boot = load_bootstrap_articles()
    print(f"Loaded {len(df_boot)} bootstrap articles.")

    snippets_df = snippets_from_bootstrap(df_boot)
    print(f"Generated {len(snippets_df)} snippets from bootstrap articles.")

    append_snippets(snippets_df)

    # Simple verification sample
    if not snippets_df.empty:
        print("\nSample bootstrap snippets:\n")
        print(snippets_df[["snippet_id", "title", "published"]].head().to_string(index=False))


if __name__ == "__main__":
    main()

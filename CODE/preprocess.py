import pandas as pd
import os
import re
import config
from tqdm import tqdm
import hashlib
import argparse

from utils.id_utils import make_article_id, make_snippet_id

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    t = text.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # Best-effort boilerplate stripping for common RSS/full-text artifacts.
    boilerplate_phrases = [
        "TechCrunch Desktop Logo",
        "TechCrunch Mobile Logo",
        "Toggle Mega Menu",
        "Submit Site Search",
        "Site Search",
        "Crunchboard",
        "Loading the player",
    ]
    for p in boilerplate_phrases:
        t = re.sub(re.escape(p), " ", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_hash(text: str) -> str:
    t = (text or "").encode("utf-8", errors="ignore")
    return hashlib.sha1(t).hexdigest()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # If we are not at the end, try to find the last space to avoid splitting words
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        # Prevent infinite loop if overlap is too large or no progress is made
        if start >= end:
            start = end
            
    return chunks

def process_articles(rebuild: bool = False):
    """Load raw articles, clean, chunk, and append snippets incrementally."""
    if not os.path.exists(config.ARTICLES_FILE):
        print("No articles file found.")
        return

    df = pd.read_parquet(config.ARTICLES_FILE)
    print(f"Loaded {len(df)} articles.")

    existing_df = pd.DataFrame()
    existing_ids = set()
    existing_keys = set()

    if (not rebuild) and os.path.exists(config.SNIPPETS_FILE):
        existing_df = pd.read_parquet(config.SNIPPETS_FILE)
        if not existing_df.empty:
            for row in existing_df.itertuples(index=False):
                article_id = getattr(row, "article_id", "") or make_article_id(
                    getattr(row, "link", ""),
                    getattr(row, "source", ""),
                    getattr(row, "title", ""),
                    getattr(row, "published", ""),
                )
                if article_id:
                    existing_keys.add((article_id, getattr(row, "chunk_index", -1)))
                snippet_id = getattr(row, "snippet_id", None)
                if snippet_id:
                    existing_ids.add(snippet_id)
        print(f"Found {len(existing_ids)} existing snippets.")

    snippets = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Articles"):
        text = clean_text(row.get("text", "") or row.get("summary", ""))
        if not text:
            continue

        article_id = row.get("article_id") or make_article_id(
            row.get("canonical_url") or row.get("link", ""),
            row.get("source", "Unknown"),
            row.get("title", ""),
            row.get("published", ""),
        )
        if not article_id:
            continue

        chunks = chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            snippet_id = make_snippet_id(article_id, i)
            if snippet_id in existing_ids or (article_id, i) in existing_keys:
                continue

            snippets.append({
                "snippet_id": snippet_id,
                "article_id": article_id,
                "source": row.get("source", "Unknown"),
                "title": row.get("title", ""),
                "link": row.get("link", ""),
                "text": chunk,
                "text_hash": text_hash(chunk),
                "published": row.get("published", ""),
                "chunk_index": i,
                "parent_fetched_at": row.get("fetched_at", "")
            })

    if not snippets and not rebuild:
        print("No new snippets generated.")
        return

    new_snippets_df = pd.DataFrame(snippets)

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if rebuild:
        combined_df = new_snippets_df
        print(f"Rebuilt snippets store with {len(combined_df)} rows.")
    elif not existing_df.empty:
        combined_df = pd.concat([existing_df, new_snippets_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["snippet_id"])
        print(f"Existing snippets: {len(existing_df)}, new: {len(new_snippets_df)}, final: {len(combined_df)}")
    else:
        combined_df = new_snippets_df
        print(f"Created new snippets store with {len(combined_df)} rows.")

    combined_df.to_parquet(config.SNIPPETS_FILE, index=False)
    print(f"Saved snippets to {config.SNIPPETS_FILE}")

    # Verification Output
    print("\n--- Sample Output (Preprocessing) ---")
    print(new_snippets_df[['snippet_id', 'text']].head(3).to_string())
    print("-------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Preprocess articles into deterministic snippets.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild snippets.parquet from scratch (overwrites file).")
    args = parser.parse_args()

    process_articles(rebuild=bool(args.rebuild))

if __name__ == "__main__":
    main()

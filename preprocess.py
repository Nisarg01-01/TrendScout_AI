"""
preprocess.py
--------------
Cleans and normalizes AI startup article text, splits into ~120-word snippets,
saves output to data/snippets.parquet, and displays sample snippets for review.
"""

import os
import re
import pandas as pd
from pathlib import Path
import random

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

RAW_PATH = Path("data/articles_raw.parquet")
OUTPUT_PATH = Path("data/snippets.parquet")
PREVIEW_PATH = Path("data/snippets_preview.csv")

# -------------------------------------------------------------------
# TEXT CLEANING UTILITIES
# -------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace, remove URLs, non-ASCII characters, and boilerplate."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)                       # remove URLs
    text = re.sub(r"[\r\n\t]+", " ", text)                    # normalize spaces
    text = re.sub(r"[^A-Za-z0-9.,!?$%&()'\"\- ]+", " ", text)  # escape hyphen safely
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def split_to_snippets(text: str, words_per_chunk: int = 120):
    """Split a long text into overlapping snippets of about 100–150 words."""
    words = text.split()
    if not words:
        return []
    snippets = []
    step = int(words_per_chunk * 0.75)  # 25% overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + words_per_chunk])
        if len(chunk.split()) >= 30:  # skip too small chunks
            snippets.append(chunk)
    return snippets


# -------------------------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------------------------

def main():
    if not RAW_PATH.exists():
        print(f"File not found: {RAW_PATH}")
        return

    os.makedirs("data", exist_ok=True)
    df = pd.read_parquet(RAW_PATH)
    print(f"Loaded {len(df)} raw articles.")

    records = []
    for _, row in df.iterrows():
        cleaned = clean_text(row.get("summary", ""))
        snippets = split_to_snippets(cleaned)
        for i, snip in enumerate(snippets):
            records.append({
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "link": row.get("link", ""),
                "published": row.get("published", ""),
                "snippet_id": f"{hash(row.link)}_{i}",
                "snippet_text": snip
            })

    out_df = pd.DataFrame(records)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Created {len(out_df)} cleaned snippets → {OUTPUT_PATH}")

    # Save preview for quick inspection
    preview = out_df.sample(min(5, len(out_df)), random_state=42)
    preview.to_csv(PREVIEW_PATH, index=False)
    print(f"Saved sample preview → {PREVIEW_PATH}\n")

    # Display sample in console
    print("Sample snippets:")
    for idx, row in preview.iterrows():
        print(f"\nTitle: {row['title']}\nSnippet:\n{row['snippet_text'][:500]}...")
        print("-" * 80)


# -------------------------------------------------------------------

if __name__ == "__main__":
    main()

import pandas as pd
import os
import uuid
import re
import config
from tqdm import tqdm

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

def process_articles():
    """Load raw articles, clean, chunk, and append snippets incrementally."""
    if not os.path.exists(config.ARTICLES_FILE):
        print("No articles file found.")
        return

    df = pd.read_parquet(config.ARTICLES_FILE)
    print(f"Loaded {len(df)} articles.")

    snippets = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Articles"):
        text = clean_text(row.get("summary", ""))
        if not text:
            continue

        chunks = chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            snippets.append({
                "snippet_id": str(uuid.uuid4()),
                "source": row.get("source", "Unknown"),
                "title": row.get("title", ""),
                "link": row.get("link", ""),
                "text": chunk,
                "published": row.get("published", ""),
                "chunk_index": i,
                "parent_fetched_at": row.get("fetched_at", "")
            })

    if not snippets:
        print("No snippets generated.")
        return

    new_snippets_df = pd.DataFrame(snippets)

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(config.SNIPPETS_FILE):
        existing_df = pd.read_parquet(config.SNIPPETS_FILE)
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
    process_articles()

if __name__ == "__main__":
    main()

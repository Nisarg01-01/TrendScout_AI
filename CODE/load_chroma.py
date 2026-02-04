import os
import pandas as pd
import chromadb
from chromadb.config import Settings
import config
from tqdm import tqdm
import numpy as np
import argparse

def load_parquet_to_chroma(*, rebuild: bool = False):
    """
    Loads the pre-computed embeddings from Parquet into a local ChromaDB.
    This allows for fast, persistent vector search without keeping everything in memory.
    """
    parquet_path = os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found. Please unzip the shared data first.")
        return

    print(f"Loading embeddings from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if df.empty:
        print("Parquet file is empty.")
        return

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    
    if rebuild:
        # Delete existing collection if we want a fresh start.
        try:
            client.delete_collection("trendscout_snippets")
            print("Deleted existing collection to ensure fresh load.")
        except Exception:
            pass  # Collection didn't exist

    collection = client.get_or_create_collection(name="trendscout_snippets")

    print(f"Inserting {len(df)} documents into ChromaDB...")
    
    # Batch insert to avoid memory issues
    batch_size = 100
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in tqdm(range(total_batches), desc="Populating Chroma"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        ids = [str(row['snippet_id']) for _, row in batch.iterrows()]
        documents = [row['text'] for _, row in batch.iterrows()]
        
        # Metadatas
        metadatas = []
        for _, row in batch.iterrows():
            meta = {
                "source": row['source'],
                "published": str(row['published']),
                "title": row['title'] if pd.notna(row['title']) else "Unknown",
                # Optional but useful for citations / UI linking.
                "link": row.get("link", "") if pd.notna(row.get("link", "")) else "",
                "article_id": str(row.get("article_id", "")) if pd.notna(row.get("article_id", "")) else "",
                "snippet_id": str(row.get("snippet_id", "")) if pd.notna(row.get("snippet_id", "")) else "",
            }
            metadatas.append(meta)

        # Embeddings - ensure they are lists of floats
        embeddings = []
        for _, row in batch.iterrows():
            emb = row['embedding']
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            embeddings.append(emb)

        # Prefer upsert for incremental runs (avoids errors on existing IDs).
        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            try:
                collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            except Exception as e:
                raise SystemExit(
                    f"Chroma collection does not support upsert and add() failed (likely due to duplicate IDs): {e}. "
                    "Re-run with --rebuild to recreate the collection."
                )

    print(f"Successfully loaded {len(df)} snippets into ChromaDB at {config.CHROMA_DB_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load snippet embeddings parquet into ChromaDB.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the Chroma collection before loading embeddings.",
    )
    args = parser.parse_args()
    load_parquet_to_chroma(rebuild=bool(args.rebuild))

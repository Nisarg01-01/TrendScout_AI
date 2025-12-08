import os
import pandas as pd
import chromadb
from chromadb.config import Settings
import config
from tqdm import tqdm
import numpy as np

def load_parquet_to_chroma():
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
    chroma_dir = os.path.join(os.getcwd(), "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Delete existing collection if we want a fresh start (optional, but good for consistency)
    try:
        client.delete_collection("trendscout_snippets")
        print("Deleted existing collection to ensure fresh load.")
    except Exception:
        pass # Collection didn't exist

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
                "title": row['title'] if pd.notna(row['title']) else "Unknown"
            }
            metadatas.append(meta)

        # Embeddings - ensure they are lists of floats
        embeddings = []
        for _, row in batch.iterrows():
            emb = row['embedding']
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            embeddings.append(emb)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"Successfully loaded {len(df)} snippets into ChromaDB at {chroma_dir}")

if __name__ == "__main__":
    load_parquet_to_chroma()

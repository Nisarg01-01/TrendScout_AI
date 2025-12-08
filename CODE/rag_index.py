import os
import json
import pandas as pd
import ollama
import config
import numpy as np

class RAGIndexer:
    def __init__(self):
        self.output_file = os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")

import os
import json
import pandas as pd
import ollama
import config
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class RAGIndexer:
    def __init__(self):
        self.output_file = os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")

    def generate_embedding(self, text):
        """Generate vector embedding for text using Ollama."""
        
        try:
            response = ollama.embeddings(model=config.LLM_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            # print(f"Embedding failed: {e}")
            return None

    def index_snippets(self):
        """Generate embeddings for new snippets and save to parquet file."""
        
        if not os.path.exists(config.SNIPPETS_FILE):
            print("No snippets file found.")
            return

        df = pd.read_parquet(config.SNIPPETS_FILE)
        print(f"Loaded {len(df)} snippets.")
        
        # Skip snippets that already have embeddings
        existing_ids = set()
        existing_df = pd.DataFrame()
        
        if os.path.exists(self.output_file):
            try:
                existing_df = pd.read_parquet(self.output_file)
                if 'snippet_id' in existing_df.columns:
                    existing_ids = set(existing_df['snippet_id'].unique())
                    print(f"Found {len(existing_ids)} already indexed snippets.")
            except Exception as e:
                print(f"Could not read existing embeddings: {e}")

        # Only index new snippets
        df_new = df[~df['snippet_id'].isin(existing_ids)].copy()
        
        if df_new.empty:
            print("All snippets are already indexed.")
            return

        print(f"Indexing {len(df_new)} new snippets...")
        
        embeddings = [None] * len(df_new)
        
        # Generate embeddings in parallel for performance
        MAX_WORKERS = 4
        
        # Submit embedding tasks to thread pool
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for _, row in df_new.iterrows():
                futures.append(executor.submit(self.generate_embedding, row['text']))
            
            for future in tqdm(futures, desc="Generating Embeddings"):
                results.append(future.result())
                
        df_new['embedding'] = results
        
        # Drop rows with failed embeddings
        df_new = df_new.dropna(subset=['embedding'])
        
        # Merge new embeddings with existing data
        if not existing_df.empty:
            final_df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            final_df = df_new
        
        print(f"Saving {len(final_df)} embeddings to {self.output_file}...")
        final_df.to_parquet(self.output_file)
        print("Indexing complete.")
        
        # Verification Output
        print("\n--- Sample Output (RAG Indexing) ---")
        print(f"Total Embeddings: {len(final_df)}")
        if not final_df.empty:
            print(f"Embedding Dimension: {len(final_df.iloc[0]['embedding'])}")
        print("------------------------------------\n")

if __name__ == "__main__":
    indexer = RAGIndexer()
    indexer.index_snippets()

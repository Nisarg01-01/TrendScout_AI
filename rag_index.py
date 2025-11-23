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
        """
        I'm asking Ollama to turn this text into a vector (a list of numbers).
        This way, we can compare the meaning of articles mathematically.
        """
        try:
            response = ollama.embeddings(model=config.LLM_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            # print(f"Embedding failed: {e}")
            return None

    def index_snippets(self):
        """
        Time to build our search index.
        I'll go through the new snippets, generate embeddings for them, and save everything to a parquet file.
        """
        if not os.path.exists(config.SNIPPETS_FILE):
            print("No snippets file found.")
            return

        df = pd.read_parquet(config.SNIPPETS_FILE)
        print(f"Loaded {len(df)} snippets.")
        
        # I don't want to re-calculate embeddings for things we've already done.
        # So I'll check what's already in our output file.
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

        # Only keep the ones we haven't seen before.
        df_new = df[~df['snippet_id'].isin(existing_ids)].copy()
        
        if df_new.empty:
            print("All snippets are already indexed.")
            return

        print(f"Indexing {len(df_new)} new snippets...")
        
        embeddings = [None] * len(df_new)
        
        # Generating embeddings can be slow, so I'll do a few at a time in parallel.
        MAX_WORKERS = 4
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(self.generate_embedding, row['text']): idx 
                for idx, row in df_new.iterrows()
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(df_new), desc="Generating Embeddings"):
                idx = future_to_idx[future]
                # Map back to the list index (which is 0 to len(df_new)-1)
                # Wait, df_new.iterrows() returns the original index. 
                # We need to align it correctly.
                # Let's just store results in a dict and map back.
                pass

        # I'm submitting all the tasks to the thread pool and collecting the results as they finish.
        results = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for _, row in df_new.iterrows():
                futures.append(executor.submit(self.generate_embedding, row['text']))
            
            for future in tqdm(futures, desc="Generating Embeddings"):
                results.append(future.result())
                
        df_new['embedding'] = results
        
        # If any embeddings failed (returned None), we'll drop those rows.
        df_new = df_new.dropna(subset=['embedding'])
        
        # Now I'll merge the new embeddings with the old ones and save the whole lot.
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

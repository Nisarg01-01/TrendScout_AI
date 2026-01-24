import os
import pandas as pd
import ollama
import config
import shutil
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class RAGIndexer:
    def __init__(self):
        self.output_file = os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")

    def _backup_embeddings(self) -> str | None:
        if not os.path.exists(self.output_file):
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = f"{self.output_file}.bak_{stamp}"
        try:
            shutil.copy2(self.output_file, bak)
            return bak
        except Exception:
            return None

    def generate_embedding(self, text: str):
        """Generate vector embedding for text using Ollama."""
        try:
            response = ollama.embeddings(model=config.LLM_MODEL, prompt=text)
            return response['embedding']
        except Exception:
            return None

    def index_snippets(self):
        """Generate embeddings for new snippets and save to parquet file."""
        if not os.path.exists(config.SNIPPETS_FILE):
            print("No snippets file found.")
            return

        df = pd.read_parquet(config.SNIPPETS_FILE)
        print(f"Loaded {len(df)} snippets.")
        current_ids = set(df["snippet_id"].astype(str)) if "snippet_id" in df.columns else set()

        existing_ids = set()
        existing_df = pd.DataFrame()

        if os.path.exists(self.output_file):
            try:
                existing_df = pd.read_parquet(self.output_file)
                if 'snippet_id' in existing_df.columns:
                    existing_ids = set(existing_df['snippet_id'].astype(str).unique())
                    stale = bool(current_ids) and any(sid not in current_ids for sid in existing_ids)
                    if stale:
                        bak = self._backup_embeddings()
                        if bak:
                            print(f"[WARN] Existing embeddings are stale vs current snippets. Backup written: {bak}")
                        existing_ids = set()
                        existing_df = pd.DataFrame()
                        print("[INFO] Rebuilding embeddings from scratch to avoid mixing old snippet_ids.")
                    else:
                        print(f"Found {len(existing_ids)} already indexed snippets.")
            except Exception as e:
                print(f"Could not read existing embeddings: {e}")

        df_new = df[~df['snippet_id'].astype(str).isin(existing_ids)].copy()

        if df_new.empty:
            print("All snippets are already indexed.")
            return

        print(f"Indexing {len(df_new)} new snippets...")

        MAX_WORKERS = 4
        embeddings = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for emb in tqdm(
                executor.map(self.generate_embedding, df_new['text']),
                total=len(df_new),
                desc="Generating Embeddings"
            ):
                embeddings.append(emb)

        df_new['embedding'] = embeddings
        df_new = df_new.dropna(subset=['embedding'])

        if not existing_df.empty:
            final_df = pd.concat([existing_df, df_new], ignore_index=True)
        else:
            final_df = df_new

        print(f"Saving {len(final_df)} embeddings to {self.output_file}...")
        final_df.to_parquet(self.output_file, index=False)
        print("Indexing complete.")

        print("\n--- Sample Output (RAG Indexing) ---")
        print(f"Total Embeddings: {len(final_df)}")
        if not final_df.empty:
            print(f"Embedding Dimension: {len(final_df.iloc[0]['embedding'])}")
        print("------------------------------------\n")


if __name__ == "__main__":
    indexer = RAGIndexer()
    indexer.index_snippets()
